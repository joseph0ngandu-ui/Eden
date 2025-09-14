import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import QtGraphicalEffects 1.15
import Eden 1.0

ApplicationWindow {
    id: mainWindow
    
    title: "Eden Trading System"
    width: 1600
    height: 1000
    minimumWidth: 1200
    minimumHeight: 800
    visible: true
    
    // Eden Dark Theme Colors
    readonly property color backgroundColor: "#0D1117"
    readonly property color surfaceColor: "#161B22"
    readonly property color cardColor: "#21262D"
    readonly property color accentGreen: "#238636"
    readonly property color accentGreenHover: "#2EA043"
    readonly property color textPrimary: "#F0F6FC"
    readonly property color textSecondary: "#8B949E"
    readonly property color textTertiary: "#656D76"
    readonly property color borderColor: "#30363D"
    readonly property color dangerColor: "#F85149"
    readonly property color warningColor: "#D29922"
    readonly property color successColor: "#3FB950"
    
    // Animation properties
    readonly property int animationDuration: 250
    readonly property int fastAnimationDuration: 150
    readonly property alias easing: Easing.OutCubic
    
    color: backgroundColor
    
    // Main layout container
    Rectangle {
        anchors.fill: parent
        color: backgroundColor
        
        RowLayout {
            anchors.fill: parent
            spacing: 0
            
            // Left Sidebar
            SideBar {
                id: sideBar
                Layout.fillHeight: true
                Layout.preferredWidth: 280
                Layout.minimumWidth: 260
                Layout.maximumWidth: 320
                
                onProjectSelected: function(projectId) {
                    contentStack.currentIndex = 0
                    projectView.loadProject(projectId)
                }
                
                onDatasetSelected: function(datasetId) {
                    contentStack.currentIndex = 1
                    datasetView.loadDataset(datasetId)
                }
                
                onBacktestSelected: function(backtestId) {
                    contentStack.currentIndex = 2
                    backtestView.loadBacktest(backtestId)
                }
            }
            
            // Main Content Area
            Rectangle {
                Layout.fillWidth: true
                Layout.fillHeight: true
                color: surfaceColor
                
                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 1
                    spacing: 1
                    
                    // Top Content Area (Charts + Right Drawer)
                    RowLayout {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.preferredHeight: parent.height * 0.65
                        spacing: 1
                        
                        // Center Chart Canvas Area
                        ChartCanvas {
                            id: chartCanvas
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            Layout.preferredWidth: parent.width * 0.75
                            
                            // Connect to Eden app data
                            backtestManager: edenApp?.backtestManager
                        }
                        
                        // Right Drawer (Parameters, GPU, Controls)
                        RightDrawer {
                            id: rightDrawer
                            Layout.fillHeight: true
                            Layout.preferredWidth: collapsed ? 60 : 350
                            Layout.maximumWidth: 400
                            Layout.minimumWidth: 60
                            
                            // Connect to managers
                            gpuManager: edenApp?.gpuManager
                            backtestManager: edenApp?.backtestManager
                            
                            Behavior on Layout.preferredWidth {
                                NumberAnimation {
                                    duration: animationDuration
                                    easing.type: easing
                                }
                            }
                        }
                    }
                    
                    // Bottom Pane (Logs, Trades, Equity)
                    BottomPane {
                        id: bottomPane
                        Layout.fillWidth: true
                        Layout.preferredHeight: collapsed ? 40 : parent.height * 0.35
                        Layout.minimumHeight: 40
                        Layout.maximumHeight: parent.height * 0.5
                        
                        // Connect to managers
                        backtestManager: edenApp?.backtestManager
                        
                        Behavior on Layout.preferredHeight {
                            NumberAnimation {
                                duration: animationDuration
                                easing.type: easing
                            }
                        }
                    }
                }
            }
        }
        
        // Status Bar Overlay
        StatusBar {
            id: statusBar
            anchors.bottom: parent.bottom
            anchors.left: parent.left
            anchors.right: parent.right
            height: 28
            
            statusText: edenApp?.statusMessage || "Loading..."
            version: edenApp?.version || "1.0.0"
            isInitialized: edenApp?.isInitialized || false
        }
        
        // Loading Overlay (shown during initialization)
        LoadingOverlay {
            id: loadingOverlay
            anchors.fill: parent
            visible: !edenApp?.isInitialized
            statusMessage: edenApp?.statusMessage || "Initializing..."
            
            Behavior on opacity {
                NumberAnimation {
                    duration: fastAnimationDuration
                    easing.type: easing
                }
            }
        }
        
        // Toast Notification System
        ToastManager {
            id: toastManager
            anchors.top: parent.top
            anchors.right: parent.right
            anchors.margins: 20
        }
    }
    
    // Stack view for different content modes
    StackLayout {
        id: contentStack
        anchors.fill: parent
        visible: false // Overlay mode, controlled by sidebar
        
        ProjectView { id: projectView }
        DatasetView { id: datasetView }
        BacktestView { id: backtestView }
    }
    
    // Window state management
    Component.onCompleted: {
        // Connect Eden app messages to toast system
        if (edenApp) {
            edenApp.messageRequested.connect(toastManager.showMessage)
            edenApp.shutdownRequested.connect(Qt.quit)
        }
        
        // Set window properties
        mainWindow.flags = Qt.Window | Qt.WindowTitleHint | Qt.WindowSystemMenuHint | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint
        
        // macOS: Use native window decorations and enable full-screen
        if (Qt.platform.os === "osx") {
            mainWindow.flags |= Qt.WindowFullscreenButtonHint
        }
        
        // Windows: Enable modern styling
        if (Qt.platform.os === "windows") {
            // Modern Windows 10/11 styling will be handled by C++ side
        }
    }
    
    onClosing: function(close) {
        if (edenApp) {
            edenApp.shutdown()
        }
        close.accepted = true
    }
    
    // Global keyboard shortcuts
    Shortcut {
        sequence: StandardKey.Quit
        onActivated: edenApp?.shutdown()
    }
    
    Shortcut {
        sequence: "Ctrl+Shift+R"
        onActivated: rightDrawer.collapsed = !rightDrawer.collapsed
    }
    
    Shortcut {
        sequence: "Ctrl+Shift+B"
        onActivated: bottomPane.collapsed = !bottomPane.collapsed
    }
    
    Shortcut {
        sequence: "F11"
        onActivated: mainWindow.visibility = mainWindow.visibility === Window.FullScreen ? Window.Windowed : Window.FullScreen
    }
}