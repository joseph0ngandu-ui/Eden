#!/bin/bash

# Eden iOS App - Quick Setup Script
# This script helps you set up the Xcode project quickly

echo "🚀 Eden iOS App - Quick Setup"
echo "================================"
echo ""

# Check if Xcode is installed
if ! command -v xcodebuild &> /dev/null; then
    echo "❌ Xcode is not installed. Please install Xcode from the App Store."
    exit 1
fi

echo "✓ Xcode found"
echo ""

# Get current directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EDEN_DIR="$SCRIPT_DIR/Eden"

# Check if Eden directory exists
if [ ! -d "$EDEN_DIR" ]; then
    echo "❌ Eden directory not found at: $EDEN_DIR"
    exit 1
fi

echo "✓ Eden source files found"
echo ""

# Count Swift files
SWIFT_FILES=$(find "$EDEN_DIR" -name "*.swift" | wc -l)
echo "📱 Found $SWIFT_FILES Swift files"
echo ""

echo "📋 Next Steps:"
echo "================================"
echo ""
echo "1. Open Xcode:"
echo "   open /Applications/Xcode.app"
echo ""
echo "2. Create New Project:"
echo "   • File → New → Project"
echo "   • Choose 'App' template"
echo "   • Product Name: Eden"
echo "   • Interface: SwiftUI"
echo "   • Language: Swift"
echo "   • Save in: $SCRIPT_DIR"
echo ""
echo "3. Add Files:"
echo "   • Drag the 'Eden' folder into your project"
echo "   • Check 'Copy items if needed'"
echo "   • Select 'Create groups'"
echo ""
echo "4. Configure Info.plist:"
echo "   • Add network permissions (see README.md)"
echo ""
echo "5. Update API endpoints:"
echo "   • Open Services/APIService.swift"
echo "   • Replace baseURL with your n8n webhook"
echo "   • Replace apiKey with your actual key"
echo ""
echo "6. Run the app:"
echo "   • Select simulator"
echo "   • Press ⌘R"
echo ""
echo "================================"
echo ""
echo "📚 For detailed instructions, see:"
echo "   $SCRIPT_DIR/README.md"
echo ""
echo "✨ Happy coding!"
