import json

def patch_03():
    path = "notebooks/03_llm_summarization.ipynb"
    with open(path, "r") as f:
        nb = json.load(f)
    for cell in nb.get("cells", []):
        if cell["cell_type"] == "code":
            src = cell["source"]
            new_src = []
            for line in src:
                if line.strip() == "from models.mistral_model import MistralModel":
                    new_src.append(line)
                    new_src.append("from models.qwen_model import QwenModel\n")
                elif line.strip() == "mistral = MistralModel()":
                    new_src.append(line)
                    new_src.append("qwen = QwenModel()\n")
                elif line.strip() == "print(\"Configured MISTRAL_MODEL:\", mistral.model_name, \"@\", mistral.base_url)":
                    new_src.append(line)
                    new_src.append("print(\"Configured QWEN_MODEL:\", qwen.model_name, \"@\", qwen.base_url)\n")
                elif line.strip() == "print(\"Mistral available:\", mistral.is_available())":
                    new_src.append(line)
                    new_src.append("print(\"Qwen available:\", qwen.is_available())\n")
                elif line.strip() == "if not mistral.is_available():":
                    new_src.append(line)
                elif line.strip() == "missing.append(mistral.model_name)":
                    new_src.append(line)
                    new_src.append("if not qwen.is_available():\n")
                    new_src.append("    missing.append(qwen.model_name)\n")
                elif line.strip() == "TRACK_B_MODEL_ID = mistral.model_name":
                    new_src.append(line)
                    new_src.append("TRACK_C_MODEL_ID = qwen.model_name\n")
                elif line.strip() == "print(\"Model id — Track B (MistralModel):\", TRACK_B_MODEL_ID)":
                    new_src.append(line)
                    new_src.append("print(\"Model id — Track C (QwenModel):\", TRACK_C_MODEL_ID)\n")
                elif "- **`.env`**: `LLM_BACKEND=vllm`" in line:
                    new_src.append(line.replace("LLAMA_MODEL`, `MISTRAL_MODEL`", "LLAMA_MODEL`, `MISTRAL_MODEL`, `QWEN_MODEL`"))
                elif "f\"## Track B (`MistralModel`) — `{TRACK_B_MODEL_ID}` via vLLM — zero-shot summary\"," in line:
                    new_src.append(line)
                    new_src.append("    f\"## Track C (`QwenModel`) — `{TRACK_C_MODEL_ID}` via vLLM — zero-shot summary\",\n")
                else:
                    new_src.append(line)
            cell["source"] = new_src
    with open(path, "w") as f:
        json.dump(nb, f, indent=2)

def patch_04():
    path = "notebooks/04_experiment_analysis.ipynb"
    with open(path, "r") as f:
        nb = json.load(f)
    for cell in nb.get("cells", []):
        if cell["cell_type"] == "markdown":
            src = cell["source"]
            new_src = []
            for line in src:
                if "research questions (LLaMA vs Mistral," in line:
                    new_src.append(line.replace("LLaMA vs Mistral,", "LLaMA vs Mistral vs Qwen,"))
                elif "LLaMA vs Mistral comparison for reports" in line:
                    new_src.append(line.replace("LLaMA vs Mistral", "LLaMA vs Mistral vs Qwen"))
                else:
                    new_src.append(line)
            cell["source"] = new_src
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)

patch_03()
patch_04()
