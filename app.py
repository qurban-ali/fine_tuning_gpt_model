import streamlit as st
import os
import tempfile
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables (for default values)
load_dotenv()

def create_fine_tuning_job(client, file_path: str, model_name: str):
    """
    Creates a fine-tuning job using the specified file and model.
    
    Args:
        client: OpenAI client instance
        file_path (str): Path to the training file
        model_name (str): Name of the model to fine-tune
        
    Returns:
        tuple: Contains uploaded file object and fine-tuning job object
    """
    # Upload the training file
    uploaded_file = client.files.create(
        file=open(file_path, "rb"),
        purpose="fine-tune"
    )

    # Start fine-tuning job with the uploaded file ID
    fine_tuning_jobs = client.fine_tuning.jobs.create(
        training_file=uploaded_file.id,
        model=model_name
    )
    
    return uploaded_file, fine_tuning_jobs

def retrieve_fine_tuning_job(client, job_id: str):
    """
    Retrieves the state of a fine-tuning job.
    
    Args:
        client: OpenAI client instance
        job_id (str): ID of the fine-tuning job to retrieve
        
    Returns:
        object: Fine-tuning job details
    """
    return client.fine_tuning.jobs.retrieve(job_id)

def cancel_fine_tuning_job(client, job_id: str):
    """
    Cancels a fine-tuning job.
    
    Args:
        client: OpenAI client instance
        job_id (str): ID of the fine-tuning job to cancel
        
    Returns:
        object: Cancelled job details
    """
    return client.fine_tuning.jobs.cancel(job_id)

# Page configuration
st.set_page_config(page_title="OpenAI Fine-Tuning Manager", layout="wide")

# Title and description
st.title("OpenAI Fine-Tuning Manager")
st.markdown("Upload your training data and fine-tune OpenAI models with a user-friendly interface.")

# API Key input (with session state to persist across reruns)
if 'api_key' not in st.session_state:
    st.session_state.api_key = os.getenv("OPENAI_API_KEY", "")

api_key = st.text_input(
    "Enter your OpenAI API Key",
    value=st.session_state.api_key,
    type="password",
    help="Your API key will not be stored permanently"
)

# Update session state when API key changes
if api_key != st.session_state.api_key:
    st.session_state.api_key = api_key

# Initialize OpenAI client if API key is provided
client = None
if st.session_state.api_key:
    try:
        client = OpenAI(api_key=st.session_state.api_key)
        st.success("API key accepted. You can now use the fine-tuning features.")
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {str(e)}")

# Available models for fine-tuning
available_models = [
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4o-2024-08-06",
    "gpt-4o-mini-2024-07-18",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0613"
]

# Model selection
selected_model = st.selectbox(
    "Select a model to fine-tune",
    options=available_models,
    index=3,
    disabled=client is None,
    help="Choose the base model you want to fine-tune"
)

# File upload section
uploaded_file = st.file_uploader(
    "Upload your training data (JSONL format)",
    type=["jsonl"],
    disabled=client is None,
    help="Upload a JSONL file with your training examples"
)

# Display file preview if uploaded
if uploaded_file is not None:
    # Read and display first few lines of the file
    file_contents = uploaded_file.getvalue().decode("utf-8")
    lines = file_contents.split("\n")
    preview_lines = lines[:5]  # Show first 5 lines
    
    st.subheader("File Preview")
    for line in preview_lines:
        st.code(line, language="json")
    
    if len(lines) > 5:
        st.info(f"... and {len(lines) - 5} more lines")

# Create fine-tuning job section
if st.button("Start Fine-Tuning", disabled=client is None or uploaded_file is None):
    try:
        with st.spinner("Starting fine-tuning job..."):
            # Create a temporary file with the uploaded content
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl", mode='wb') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Create fine-tuning job using local function
                uploaded_file_obj, fine_tuning_job = create_fine_tuning_job(
                    client=client,
                    file_path=tmp_file_path,
                    model_name=selected_model
                )
                
                # Clear the file uploader state to prevent duplicate uploads
                uploaded_file.close()
                st.session_state.uploaded_file = None
                
                # Display job information
                st.success(f"Fine-tuning job created successfully!")
                st.json({
                    "job_id": fine_tuning_job.id,
                    "status": fine_tuning_job.status,
                    "model": fine_tuning_job.model,
                    "file_id": uploaded_file_obj.id
                })
                
                # Store job ID in session state
                if 'jobs' not in st.session_state:
                    st.session_state.jobs = []
                
                st.session_state.jobs.append({
                    "job_id": fine_tuning_job.id,
                    "status": fine_tuning_job.status,
                    "model": fine_tuning_job.model,
                    "created_at": fine_tuning_job.created_at
                })
                
            finally:
                # Clean up the temporary file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
                
    except Exception as e:
        st.error(f"Error creating fine-tuning job: {str(e)}")

# Job management section
st.subheader("Manage Fine-Tuning Jobs")

# Input for job ID to check status
job_id = st.text_input(
    "Enter a job ID to check status",
    disabled=client is None,
    help="Enter the ID of an existing fine-tuning job"
)

# Check job status button
if st.button("Check Job Status", disabled=client is None or not job_id):
    try:
        with st.spinner("Retrieving job status..."):
            # Retrieve job status using local function
            job = retrieve_fine_tuning_job(client=client, job_id=job_id)
            
            # Display job information
            st.json({
                "job_id": job.id,
                "status": job.status,
                "model": job.model,
                "fine_tuned_model": job.fine_tuned_model if job.fine_tuned_model else "Not available yet"
            })
    except Exception as e:
        st.error(f"Error retrieving job status: {str(e)}")

# Cancel job button
if st.button("Cancel Job", disabled=client is None or not job_id):
    try:
        with st.spinner("Cancelling job..."):
            # Cancel job using local function
            cancelled_job = cancel_fine_tuning_job(client=client, job_id=job_id)
            
            # Display cancellation information
            st.success(f"Job {job_id} cancelled successfully!")
            st.json({
                "job_id": cancelled_job.id,
                "status": cancelled_job.status
            })
    except Exception as e:
        st.error(f"Error cancelling job: {str(e)}")

# Display recent jobs from session state
if 'jobs' in st.session_state and st.session_state.jobs:
    st.subheader("Recent Jobs")
    for job in st.session_state.jobs:
        with st.expander(f"Job: {job['job_id']}"):
            st.json(job)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and OpenAI API")
