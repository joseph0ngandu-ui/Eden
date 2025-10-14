# Eden Auto-Push Setup Script
# This script sets up automatic git pushing for the Eden trading system

param(
    [string]$CommitMessage = "Auto-update from Eden development session"
)

# Colors for output
$Green = [System.ConsoleColor]::Green
$Yellow = [System.ConsoleColor]::Yellow
$Red = [System.ConsoleColor]::Red
$Cyan = [System.ConsoleColor]::Cyan

function Write-ColoredOutput {
    param($Message, $Color = [System.ConsoleColor]::White)
    Write-Host $Message -ForegroundColor $Color
}

Write-ColoredOutput "🚀 Eden Auto-Push Setup" $Cyan
Write-ColoredOutput "======================" $Cyan
Write-ColoredOutput ""

# Check if we're in a git repository
if (-not (Test-Path .git)) {
    Write-ColoredOutput "❌ Not in a git repository!" $Red
    exit 1
}

# Check for changes
Write-ColoredOutput "📊 Checking for changes..." $Yellow
$status = git status --porcelain

if ([string]::IsNullOrEmpty($status)) {
    Write-ColoredOutput "✅ No changes to commit" $Green
    exit 0
}

Write-ColoredOutput "📝 Changes detected:" $Yellow
git status --short

Write-ColoredOutput ""
Write-ColoredOutput "🔄 Adding all changes..." $Yellow
git add .

Write-ColoredOutput "💾 Committing changes..." $Yellow
git commit -m $CommitMessage

Write-ColoredOutput "🌐 Pushing to GitHub..." $Yellow
try {
    git push origin main
    Write-ColoredOutput "✅ Successfully pushed to GitHub!" $Green
} catch {
    # Try master branch if main fails
    Write-ColoredOutput "⚠️ 'main' branch failed, trying 'master'..." $Yellow
    try {
        git push origin master
        Write-ColoredOutput "✅ Successfully pushed to GitHub (master branch)!" $Green
    } catch {
        Write-ColoredOutput "❌ Failed to push to GitHub. You may need to authenticate first." $Red
        Write-ColoredOutput "💡 Try: gh auth login" $Yellow
        exit 1
    }
}

Write-ColoredOutput ""
Write-ColoredOutput "🎉 Auto-push completed successfully!" $Green