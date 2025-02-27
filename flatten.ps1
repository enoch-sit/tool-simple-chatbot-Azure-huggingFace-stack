# Define paths
$sourceFolder = "." # Current directory, change as needed
$outputFile = "combined_files.txt"
$gitignoreFile = "$sourceFolder\.gitignore"

# Function to check if a path matches any pattern in .gitignore
function Test-GitignoreMatch {
    param (
        [string]$Path,
        [string[]]$IgnorePatterns
    )

    $relativePath = $Path.Replace("$sourceFolder\", "").Replace("\", "/")
    
    foreach ($pattern in $IgnorePatterns) {
        # Skip empty lines and comments
        if ([string]::IsNullOrWhiteSpace($pattern) -or $pattern.StartsWith("#")) {
            continue
        }
        
        # Remove leading/trailing whitespace
        $pattern = $pattern.Trim()
        
        # Handle negation patterns (not supported in this simple implementation)
        if ($pattern.StartsWith("!")) {
            continue
        }
        
        # Handle directory-specific patterns
        $dirPattern = $pattern.EndsWith("/")
        if ($dirPattern) {
            $pattern = $pattern.TrimEnd("/")
        }
        
        # Convert .gitignore pattern to regex
        $regex = $pattern.Replace(".", "\.").Replace("*", ".*").Replace("?", ".") + $(if ($dirPattern) { "(/.*|)$" } else { "$" })
        
        if ($relativePath -match $regex) {
            return $true
        }
    }
    
    return $false
}

# Read .gitignore file
$ignorePatterns = @()
if (Test-Path $gitignoreFile) {
    $ignorePatterns = Get-Content $gitignoreFile
}

# Add the output file itself to ignored patterns
$ignorePatterns += $outputFile

# Create or clear the output file
"" | Set-Content $outputFile

# Process all files
Get-ChildItem -Path $sourceFolder -Recurse -File | ForEach-Object {
    # Check if file should be ignored
    if (-not (Test-GitignoreMatch -Path $_.FullName -IgnorePatterns $ignorePatterns)) {
        # Add file path and name as header
        "===================================================" | Add-Content $outputFile
        "FILE: $($_.FullName)" | Add-Content $outputFile
        "===================================================" | Add-Content $outputFile
        
        # Add file content
        Get-Content $_.FullName | Add-Content $outputFile
        
        # Add empty line for separation
        "" | Add-Content $outputFile
    }
}

Write-Host "All files have been combined into $outputFile"