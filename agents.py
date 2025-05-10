from openai import OpenAI
import os
import re
import logging
from src.lean_runner import execute_lean_code

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class LLM_Agent:
    """
    Base agent class for interacting with language models.
    """
    def __init__(self, model: str = "gpt-4o"):
        self.model = model

    def makeUnitTest(self, question_text: str, function_name) -> str:
        unitTests = self.query(
            f"Given the function description: \n {question_text} \n\n make unit test "
            f"with the format:\n #guard f{function_name} [input] [output]"
        )
        return unitTests

    def query(self, prompt: str, system_prompt: str = None) -> str:
        return self.chat([
            {"role": "system", "content": system_prompt or "Implement the requested query relating to Lean4 precisely"},
            {"role": "user", "content": prompt}
        ])

    def chat(self, messages: list[dict]) -> str:
        completion = client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return completion.choices[0].message.content.strip()

class Reasoning_Agent(LLM_Agent):
    """
    Agent responsible for generating and verifying Lean 4 code and proofs.
    """
    def __init__(self, question_text: str, skeleton: str, agent: LLM_Agent, model: str = "o3-mini"):
        super().__init__(model=model)
        self.question = question_text
        self.skeleton = skeleton
        self.agent = agent
        self.code = ""
        self.proof = ""
        self.unitTest = ""
        self.functions = {}  # Will store example functions
    
    def parse_lean_file_improved(self, file_content):
        """
        Parse a Lean file and extract functions with their implementations.
        More robust implementation that handles indentation and nested code blocks.
        
        Args:
            file_content (str): Content of the Lean file
        
        Returns:
            dict: Dictionary with function names as keys and their implementations as values
        """
        # Split the file by the delimiter to separate different function blocks
        sections = file_content.split("=====================================================================")
        
        self.functions = {}
        
        for section in sections:
            if not section.strip():
                continue
            
            # Skip initial imports
            lines = [line for line in section.strip().split('\n') if not line.strip().startswith('import')]
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # Check for function definition
                if line.startswith('def '):
                    parts = line.split(' ', 2)
                    if len(parts) >= 2:
                        # Extract function name, handling parameters in the name
                        function_name = parts[1].split('(')[0].split(':')[0].strip()
                        
                        # Skip spec functions
                        if '_spec' in function_name:
                            i += 1
                            continue
                        
                        # Collect the implementation
                        implementation = [line]
                        
                        # Find the end of this function (next def or theorem)
                        j = i + 1
                        while j < len(lines):
                            next_line = lines[j].strip()
                            if (next_line.startswith('def ') or 
                                next_line.startswith('theorem')):
                                break
                            if next_line:  # Skip empty lines in output
                                implementation.append(lines[j])
                            j += 1
                        
                        # Store the function
                        self.functions[function_name] = '\n'.join(implementation)
                        logger.info(f"Loaded function: {function_name}")
                        
                        # Move to the next function
                        i = j - 1
                
                i += 1
        
        logger.info(f"Total functions loaded: {len(self.functions)}")
        logger.info(f"Available functions: {', '.join(self.functions.keys())}")

    def extract_problem_details(self) -> dict[str, str]:
        details = {}

        fn_match = re.search(r'def\s+(\w+)', self.skeleton)
        if fn_match:
            details["function_name"] = fn_match.group(1)
            logger.info(f"Extracted function name: {details['function_name']}")

        type_match = re.search(r'def\s+\w+\s+\((.*?)\)\s*:\s*(.*?):=', self.skeleton)
        if type_match:
            details["params"] = type_match.group(1)
            details["return_type"] = type_match.group(2)
            logger.info(f"Extracted parameters: {details['params']}")
            logger.info(f"Extracted return type: {details['return_type']}")

        spec_match = re.search(r'def\s+(\w+_spec)\s*\(.*?\)\s*:\s*Prop\s*:=\s*\n\s*-- << SPEC START >>\s*(.*?)\s*-- << SPEC END >>', self.skeleton, re.DOTALL)
        if spec_match:
            details["spec_name"] = spec_match.group(1)
            details["spec"] = spec_match.group(2).strip()
            logger.info(f"Extracted spec name: {details['spec_name']}")
            logger.info(f"Extracted spec: {details['spec']}")

        theorem_match = re.search(r'theorem\s+(\w+)', self.skeleton)
        if theorem_match:
            details["theorem_name"] = theorem_match.group(1)
            logger.info(f"Extracted theorem name: {details['theorem_name']}")

        self.unitTest = self.agent.makeUnitTest(self.question, details["function_name"])
        logger.info("Generated unit test")

        return details

    def generate_code(self, additional_message="") -> str:
        logger.info("Generating implementation code...")
        details = self.extract_problem_details()
        prompt = f"""
            Given the following Lean 4 problem:

            {self.question}

            And the skeleton:

            {self.skeleton}

            Please provide only the implementation code that would replace {{{{code}}}} in:

            def {details.get('function_name', 'function')} ({details.get('params', '')}) : {details.get('return_type', '')} := -- << CODE START >> {{{{code}}}} -- << CODE END >>
            
            {f"Last implementation had this error: {additional_message}" if additional_message else ""}
            Return only the code, no explanations.
            DO NOT use markdown at all, just to code ONLY
        """
        self.code = self.query(prompt)
        logger.info("Code generation complete")
        return self.code

    def generate_proof(self) -> str:
        logger.info("Generating proof...")
        details = self.extract_problem_details()
        
        # First attempt at generating a proof or determining if we need an example
        prompt = f"""
            Given the following Lean 4 problem:

            {self.question}

            The skeleton:

            {self.skeleton}

            And the implementation:

            def {details.get('function_name', 'function')} ({details.get('params', '')}) : {details.get('return_type', '')} := 
            {self.code}

            Please provide only the proof code that would replace {{{{proof}}}} in:

            theorem {details.get('theorem_name', 'theorem')} : ... := by -- << PROOF START >> unfold {details.get('function_name', 'function')} {details.get('spec_name', 'spec')} {{{{proof}}}} -- << PROOF END >>

            Your main goal is {details.get('spec')}
            Directive:
            For very simple test cases, use simp, do not use rfl
            Even if it is simple but there is a very similar case, retrieve it:
                choose one most similar example our of:
                {list(self.functions.keys())}. Return the format "RETRIEVE, example_name"
            """
        
        response = self.query(prompt)
        logger.info(f"Initial proof response: {response[:50]}...")
        
        # Check if we need to retrieve an example
        if "RETRIEVE" in response:
            logger.info("Example retrieval requested")
            # Extract the example name more robustly
            match = re.search(r"RETRIEVE,\s*(\w+)", response)
            if match:
                example_name = match.group(1).strip()
                logger.info(f"Example requested: {example_name}")
                
                # Check if the example exists
                if example_name in self.functions:
                    example = self.functions[example_name]
                    logger.info(f"Found example: {example_name}")
                    
                    # Generate proof using the example
                    prompt = f"""
                    Given the following Lean 4 problem:

                    {self.question}

                    The skeleton:

                    {self.skeleton}

                    The implementation:
                    
                    def {details.get('function_name', 'function')} ({details.get('params', '')}) : {details.get('return_type', '')} := 
                    {self.code}

                    And an example:

                    {example}

                    Please provide only the proof code that would replace {{{{proof}}}} in:

                    theorem {details.get('theorem_name', 'theorem')} : ... := by -- << PROOF START >> unfold {details.get('function_name', 'function')} {details.get('spec_name', 'spec')} {{{{proof}}}} -- << PROOF END >>

                    Your main goal is {details.get('spec')}
                    
                    Do not use Array.all_iff, or Array.map, or Array.get
                    """

                    self.proof = self.query(prompt)
                    logger.info(f"Generated proof using example {example_name}")
                else:
                    logger.warning(f"Example {example_name} not found. Using generic proof instead.")
                    # Fall back to a generic proof approach if example not found
                    prompt = f"""
                    The requested example was not found. Please provide a general proof approach for:

                    theorem {details.get('theorem_name', 'theorem')} : ... := by -- << PROOF START >> unfold {details.get('function_name', 'function')} {details.get('spec_name', 'spec')} {{{{proof}}}} -- << PROOF END >>

                    Your main goal is {details.get('spec')}
                    """
                    self.proof = self.query(prompt)
            else:
                logger.warning("Example name could not be parsed from response")
                self.proof = response  # Use the original response if we can't parse the example name
        else:
            logger.info("Using direct proof generation (no example needed)")
            self.proof = response
            
        return self.proof

    def solve(self) -> dict[str, str]:
        logger.info(f"Solving problem: {self.question[:50]}...")
        
        try:
            with open("./documents/example2.txt") as examples:
                file_content = examples.read()
                logger.info(f"Example file loaded, size: {len(file_content)} bytes")
                self.parse_lean_file_improved(file_content)
        except Exception as e:
            logger.error(f"Error loading example file: {str(e)}")
            self.functions = {}  # Ensure we have an empty dict if file loading fails
        
        self.generate_code()
        logger.info("Code generation completed")
        
        self.generate_proof()
        logger.info("Proof generation completed")
        
        complete_solution = self.skeleton.replace("{{code}}", self.code).replace("{{proof}}", self.proof)
        logger.info(complete_solution)
        
        return {
            "code": self.code,
            "proof": self.proof,
            "complete_solution": complete_solution
        }