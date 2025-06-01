# Start from the Jupyter minimal notebook image
FROM jupyter/minimal-notebook

# Switch to root user to install packages
USER root

# Install the Python packages using pip
RUN pip install openai tiktoken python-dotenv

# Switch back to the jovyan user (default Jupyter user)
USER $NB_UID

# Expose the default Jupyter port
EXPOSE 8888