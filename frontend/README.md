# CAPP GPT Web Client

This is a simple web interface to interact with the CAPP GPT web service.

## How to Use

1.  **Start the backend service:**
    Make sure the main CAPP GPT service is running. You can start it from the root directory of the project with the following command:
    ```bash
    uv run serve
    ```
    The service should be available at `http://localhost:8000`.

2.  **Open the web client:**
    Open the `index.html` file in your web browser. You can do this by double-clicking the file or by using the "Open File" option in your browser's menu.

3.  **Use the interface:**
    - The form fields will be automatically populated with valid options from the backend service.
    - Select the desired part characteristics.
    - Adjust the "Max Processes" and "Temperature" settings if needed.
    - Click the "Predict" button to get the recommended manufacturing processes.
    - The results will be displayed below the form.

## Requirements

- A modern web browser that supports JavaScript's Fetch API.
- The backend service must be running and accessible at `http://localhost:8000`.

## Notes

This web client is completely decoupled from the backend service. It is a set of static files (HTML, CSS, JavaScript) that can be served from any web server or opened directly from the local filesystem.
