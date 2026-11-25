# OpenAI Fine-Tuning Manager

This application provides a user-friendly interface for fine-tuning OpenAI models using your custom training data.

## Features

- Secure API key management
- Selection from available fine-tuning models
- JSONL file upload for training data
- Job creation and management
- Status checking for existing jobs
- Job cancellation

## Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.8 or higher
- pip (Python package manager)

### Installation

Clone the repository and create an isolated Python environment:

```bash
git clone https://github.com/qurban-ali/fine_tuning_gpt_model.git
cd fine_tuning_gpt_model
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `venv\Scripts\Activate.ps1`.

### Running the Application

Start the Streamlit application with:

```bash
streamlit run app.py --server.address 127.0.0.1
```

This will open the application in your default web browser.

## Usage

1. **Enter your OpenAI API Key** - This is required to access the OpenAI API services
2. **Select a model** - Choose which base model you want to fine-tune
3. **Upload your training data** - Provide a JSONL file with your training examples
4. **Start Fine-Tuning** - Begin the fine-tuning process
5. **Manage Jobs** - Check status or cancel existing jobs

## Training Data Format

Your training data should be in JSONL format, with each line containing a valid JSON object with the following structure:

```json
{"messages": [{"role": "system", "content": "Your system message"}, {"role": "user", "content": "User message"}, {"role": "assistant", "content": "Assistant response"}]}
```

Each line represents a single training example.

## Notes

- Fine-tuning can take several hours depending on the size of your dataset and the model you choose
- The app reads `OPENAI_API_KEY` from the environment (including a local `.env` file) or accepts a key in a password field. Do not commit keys or share the local server with untrusted users.
- The model selector is a fixed list in the application; availability depends on your OpenAI project and may change.
- Uploading training data and creating jobs sends data to OpenAI and can incur charges. Use only data you are authorized to upload and remove unnecessary personal or confidential information.
- Make sure your JSONL file is properly formatted to avoid errors

## Troubleshooting

If you encounter any issues:

1. Verify your API key is correct and has the necessary permissions
2. Check that your JSONL file follows the required format
3. Ensure you have a stable internet connection

---

Made with ❤️ using Streamlit and OpenAI API
