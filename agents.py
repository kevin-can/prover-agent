from dataclasses import dataclass
import os
import subprocess
import tempfile
from openai import OpenAI

# --- define a single TaskSpec to carry *all* the bits of a problem around ---
@dataclass
class TaskSpec:
    """Encapsulates everything needed to implement & verify a Lean4 task."""
    problem: str                     # the text of the problem statement
    code_template: str               # e.g. contents of your {{code}}+{{proof}} .lean template
    proof_template: str              # (optional) if separate from code_template
    unit_tests: str                  # the full unit‐test code to append to a solve
    # any other fields you like…

# --- a generic LLM wrapper that never touches disk paths, only specs ---
class LLM_Agent:
    def __init__(self, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def get_response(self, messages) -> str:
        c = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return c.choices[0].message.content.strip()

    def implementProof(self, plan: str) -> str:
        msgs = [
            {"role": "system", "content": "Implement the following Lean4 proof plan without commentary:"},
            {"role": "user",   "content": plan},
        ]
        return self.get_response(msgs)

    def verify(self, code: str, spec: TaskSpec, timeout: int = 30) -> dict:
        """
        Verifies `code` + spec.unit_tests by invoking `lean`.
        Returns { success, output, errors }.
        """
        combined = code + "\n\n" + spec.unit_tests
        with tempfile.NamedTemporaryFile("w+", suffix=".lean", delete=False) as tmp:
            tmp.write(combined)
            path = tmp.name

        try:
            proc = subprocess.run(
                ["lean", path],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return {
                "success": proc.returncode == 0,
                "output": proc.stdout,
                "errors": proc.stderr
            }
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

# --- a reasoning agent that also never uses file‐paths, just the spec object ---
class Reasoning_Agent(LLM_Agent):
    def __init__(self, spec: TaskSpec, verifier: LLM_Agent, model: str = "o3-mini"):
        """
        spec     -- an in‑memory TaskSpec
        verifier -- another LLM_Agent to run .verify()
        """
        super().__init__(model=model)
        self.spec = spec
        self.verifier = verifier
        self.code = ""
        self.proof = ""

    def makePlanProof(self) -> str:
        msgs = [
            {"role": "system", "content": "Create a proof plan outline in Lean4 pseudocode for the following problem:"},
            {"role": "user",   "content": self.spec.problem}
        ]
        return self.get_response(msgs)

    def solveCode(self) -> str:
        if not self.code:
            msgs = [
                {"role": "system", "content": "Write a Lean4 implementation (no comments) for the following problem:"},
                {"role": "user",   "content": self.spec.problem}
            ]
        else:
            msgs = [
                {"role": "system", "content": "The previous implementation failed tests—please correct it based on these errors:"},
                {"role": "user",   "content": self.code}
            ]
        self.code = self.get_response(msgs)
        return self.code

    def makeProof(self, plan: str) -> str:
        msgs = [
            {"role": "system", "content": "Implement the following Lean4 proof plan without extra commentary:"},
            {"role": "user",   "content": plan}
        ]
        self.proof = self.get_response(msgs)
        return self.proof

    def solve(self, max_iterations: int = 3) -> dict:
        # 1) Generate or refine implementation
        impl = self.solveCode()
        impl_res = self.verifier.verify(impl, self.spec)
        if not impl_res["success"]:
            # feed errors back and retry
            fix_msg = f"Implementation errors:\n{impl_res['errors']}\nPlease correct the code."
            self.code = self.get_response([{"role":"system","content":fix_msg}])
            impl_res = self.verifier.verify(self.code, self.spec)

        # 2) Plan the proof
        plan = self.makePlanProof()

        # 3) Iteratively write & verify the proof
        proof_res = None
        for _ in range(max_iterations):
            candidate = self.makeProof(plan)
            combined = self.code + "\n\n" + candidate
            proof_res = self.verifier.verify(combined, self.spec)
            if proof_res["success"]:
                self.proof = candidate
                break
            # otherwise refine the plan
            err = proof_res["errors"]
            plan = self.get_response([
                {"role":"system",
                 "content":f"Proof failed with errors:\n{err}\nPlease update the proof plan accordingly."}
            ])

        return {
            "implementation_verify": impl_res,
            "proof_verify": proof_res,
            "final_code": self.code,
            "final_proof": self.proof
        }

# --- example of how you might wire it up in your main_workflow.py ---

def main_workflow_from_spec(spec: TaskSpec):
    llm = LLM_Agent(model="gpt-4o")
    agent = Reasoning_Agent(spec, verifier=llm, model="o3-mini")
    result = agent.solve()
    return result

# elsewhere, you read from disk once:
#   problem, code_tpl = get_problem_and_code_from_taskpath(...)
#   proof_tpl         = get_task_lean_template_from_taskpath(...)
#   tests             = get_unit_tests_from_taskpath(...)
# spec = TaskSpec(problem, code_tpl, proof_tpl, tests)
# main_workflow_from_spec(spec)
