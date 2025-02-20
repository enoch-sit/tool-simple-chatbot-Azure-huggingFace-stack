# Define the directory structure
$directories = @(
    "chatbot_project",
    "chatbot_project/backend",
    "chatbot_project/backend/main.py",
    "chatbot_project/backend/database.py",
    "chatbot_project/backend/rag_agent.py",
    "chatbot_project/frontend",
    "chatbot_project/frontend/app.py",
    "chatbot_project/.env"
    #,
    #"chatbot_project/requirements.txt"
)

# Create the directories and files
foreach ($dir in $directories) {
    $path = [System.IO.Path]::GetFullPath($dir)
    if (-not (Test-Path $path)) {
        if ($dir -match "\.\w+$") {
            # Create file
            New-Item -ItemType File -Path $path
        } else {
            # Create directory
            New-Item -ItemType Directory -Path $path
        }
    }
}
