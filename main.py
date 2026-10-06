import os
import re
import traceback
from io import StringIO
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI


# ============================================================
# FastAPI app
# ============================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Request model
# ============================================================

class CodeRequest(BaseModel):
    code: str


# ============================================================
# AI response model
# ============================================================

class ErrorAnalysis(BaseModel):
    error_lines: List[int]


# ============================================================
# Python execution tool
# ============================================================

def execute_python_code(code: str) -> dict:
    """
    Execute Python code and return exact stdout or traceback.
    """

    import sys

    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        exec(code)

        output = sys.stdout.getvalue()

        return {
            "success": True,
            "output": output
        }

    except Exception:
        output = traceback.format_exc()

        return {
            "success": False,
            "output": output
        }

    finally:
        sys.stdout = old_stdout


# ============================================================
# AI error analysis
# ============================================================

def analyze_error_with_ai(code: str, traceback_text: str) -> List[int]:
    """
    Use AI Pipe + OpenRouter to identify error line numbers.
    """

    token = os.environ.get("AIPIPE_TOKEN")

    if not token:
        raise RuntimeError("AIPIPE_TOKEN environment variable is not set.")

    # AI Pipe documentation:
    # OpenRouter-compatible endpoint
    client = OpenAI(
        base_url="https://aipipe.org/openrouter/v1",
        api_key=token
    )

    prompt = f"""
Analyze this Python code and its traceback.

Identify the exact line number(s) in the submitted Python code
where the error occurred.

Rules:
- Return only line numbers from the submitted code.
- Do not return framework or internal Python lines.
- Use the traceback as the primary evidence.
- Return JSON with the key "error_lines".
- "error_lines" must be a list of integers.

CODE:
{code}

TRACEBACK:
{traceback_text}
"""

    response = client.chat.completions.create(
        model="openai/gpt-4.1-nano",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise Python traceback analyzer. "
                    "Return only the line numbers where the submitted "
                    "Python code caused the error."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "error_analysis",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "error_lines": {
                            "type": "array",
                            "items": {
                                "type": "integer"
                            }
                        }
                    },
                    "required": ["error_lines"],
                    "additionalProperties": False
                }
            }
        }
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError("AI returned an empty response.")

    result = ErrorAnalysis.model_validate_json(content)

    # --------------------------------------------------------
    # Validate AI output against actual traceback line numbers
    # --------------------------------------------------------

    traceback_line_numbers = sorted({
        int(n)
        for n in re.findall(
            r'File "<string>", line (\d+)',
            traceback_text
        )
    })

    if traceback_line_numbers:

        valid_lines = [
            line
            for line in result.error_lines
            if line in traceback_line_numbers
        ]

        if valid_lines:
            return valid_lines

        return traceback_line_numbers

    return result.error_lines


# ============================================================
# Main endpoint
# ============================================================

@app.post("/code-interpreter")
def code_interpreter(request: CodeRequest):

    execution = execute_python_code(request.code)

    # Successful execution:
    # return exact output and DO NOT call AI.
    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"]
        }

    # Failed execution:
    # ask AI to identify the error line(s).
    error_lines = analyze_error_with_ai(
        request.code,
        execution["output"]
    )

    return {
        "error": error_lines,
        "result": execution["output"]
    }


# ============================================================
# Health check
# ============================================================

@app.get("/")
def root():
    return {"status": "ok"}