# PowerShell script to push rx-ocr-engine to GitHub

param(
    [string]$RepoUrl
)

if (-not $RepoUrl) {
    Write-Host "==========================================================" -ForegroundColor Cyan
    Write-Host "  Medicine Prescription OCR - GitHub Push Helper" -ForegroundColor Cyan
    Write-Host "==========================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Please provide your GitHub repository URL."
    Write-Host "Example: https://github.com/your-username/medicine-prescription-ocr.git"
    Write-Host ""
    $RepoUrl = Read-Host "Enter GitHub Repository URL"
}

if (-not $RepoUrl) {
    Write-Host "No URL entered. Aborting." -ForegroundColor Red
    exit 1
}

Write-Host "`n[*] Configuring Git remote..." -ForegroundColor Yellow
git remote remove origin 2>$null
git remote add origin $RepoUrl
git branch -M main

Write-Host "[*] Pushing to GitHub ($RepoUrl)..." -ForegroundColor Yellow
git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[+] Repository successfully pushed to GitHub!" -ForegroundColor Green
} else {
    Write-Host "`n[!] Push failed. Please verify your repository URL and Git credentials/token." -ForegroundColor Red
}
