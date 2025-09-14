import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtGraphicalEffects 1.15

Rectangle {
    id: sideBar
    
    color: "#0D1117"
    border.width: 1
    border.color: "#30363D"
    
    // Signals for main window
    signal projectSelected(string projectId)
    signal datasetSelected(string datasetId) 
    signal backtestSelected(string backtestId)
    
    // Animation properties
    readonly property int animationDuration: 200
    readonly property color hoverColor: "#21262D"
    readonly property color selectedColor: "#238636"
    readonly property color textColor: "#F0F6FC"
    readonly property color secondaryTextColor: "#8B949E"
    
    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 0
        spacing: 0
        
        // Header with Eden logo/title
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 60
            color: "#161B22"
            border.width: 1
            border.color: "#30363D"
            
            RowLayout {
                anchors.fill: parent
                anchors.margins: 16
                spacing: 12
                
                // Eden icon placeholder
                Rectangle {
                    Layout.preferredWidth: 28
                    Layout.preferredHeight: 28
                    radius: 6
                    color: selectedColor
                    
                    Text {
                        anchors.centerIn: parent
                        text: "E"
                        font.pixelSize: 16
                        font.bold: true
                        color: "white"
                    }
                }
                
                Text {
                    Layout.fillWidth: true
                    text: "Eden"
                    font.pixelSize: 18
                    font.bold: true
                    color: textColor
                }
                
                // Settings button
                Rectangle {
                    Layout.preferredWidth: 24
                    Layout.preferredHeight: 24
                    radius: 4
                    color: settingsMouseArea.containsMouse ? hoverColor : "transparent"
                    
                    Text {
                        anchors.centerIn: parent
                        text: "⚙"
                        font.pixelSize: 14
                        color: secondaryTextColor
                    }
                    
                    MouseArea {
                        id: settingsMouseArea
                        anchors.fill: parent
                        hoverEnabled: true
                        onClicked: console.log("Settings clicked")
                    }
                    
                    Behavior on color {
                        ColorAnimation { duration: animationDuration }
                    }
                }
            }
        }
        
        // Main sidebar content
        ScrollView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            
            contentWidth: availableWidth
            ScrollBar.horizontal.policy: ScrollBar.AlwaysOff
            ScrollBar.vertical.policy: ScrollBar.AsNeeded
            
            ColumnLayout {
                width: parent.width
                spacing: 8
                
                // Projects Section
                SidebarSection {
                    Layout.fillWidth: true
                    title: "Projects"
                    expanded: true
                    
                    content: ColumnLayout {
                        spacing: 2
                        
                        SidebarItem {
                            Layout.fillWidth: true
                            text: "XAUUSD Strategy"
                            subtitle: "ICT + ML Ensemble"
                            icon: "📊"
                            selected: false
                            onClicked: sideBar.projectSelected("xauusd-ict-ml")
                        }
                        
                        SidebarItem {
                            Layout.fillWidth: true
                            text: "Forex Scalping"
                            subtitle: "Mean Reversion"
                            icon: "⚡"
                            selected: false
                            onClicked: sideBar.projectSelected("forex-scalping")
                        }
                        
                        SidebarItem {
                            Layout.fillWidth: true
                            text: "Crypto Momentum"
                            subtitle: "Multi-timeframe"
                            icon: "🚀"
                            selected: false
                            onClicked: sideBar.projectSelected("crypto-momentum")
                        }
                        
                        // Add new project button
                        Rectangle {
                            Layout.fillWidth: true
                            Layout.preferredHeight: 36
                            Layout.margins: 8
                            color: addProjectMouse.containsMouse ? hoverColor : "transparent"
                            border.width: 1
                            border.color: "#30363D"
                            radius: 6
                            
                            RowLayout {
                                anchors.fill: parent
                                anchors.margins: 8
                                spacing: 8
                                
                                Text {
                                    text: "+"
                                    font.pixelSize: 16
                                    color: selectedColor
                                }
                                
                                Text {
                                    Layout.fillWidth: true
                                    text: "New Project"
                                    font.pixelSize: 12
                                    color: secondaryTextColor
                                }
                            }
                            
                            MouseArea {
                                id: addProjectMouse
                                anchors.fill: parent
                                hoverEnabled: true
                                onClicked: console.log("Add project clicked")
                            }
                            
                            Behavior on color {
                                ColorAnimation { duration: animationDuration }
                            }
                        }
                    }
                }
                
                // Datasets Section
                SidebarSection {
                    Layout.fillWidth: true
                    title: "Datasets"
                    expanded: true
                    
                    content: ColumnLayout {
                        spacing: 2
                        
                        SidebarItem {
                            Layout.fillWidth: true
                            text: "XAUUSD 1H"
                            subtitle: "2018-2024 • 52.4K bars"
                            icon: "📈"
                            selected: false
                            onClicked: sideBar.datasetSelected("xauusd-1h")
                        }
                        
                        SidebarItem {
                            Layout.fillWidth: true
                            text: "EURUSD 15M"
                            subtitle: "2020-2024 • 139.2K bars"
                            icon: "📊"
                            selected: false
                            onClicked: sideBar.datasetSelected("eurusd-15m")
                        }
                        
                        SidebarItem {
                            Layout.fillWidth: true
                            text: "US30 1D"
                            subtitle: "2015-2024 • 2.3K bars"
                            icon: "📉"
                            selected: false
                            onClicked: sideBar.datasetSelected("us30-1d")
                        }
                        
                        // Import dataset button
                        Rectangle {
                            Layout.fillWidth: true
                            Layout.preferredHeight: 36
                            Layout.margins: 8
                            color: importDataMouse.containsMouse ? hoverColor : "transparent"
                            border.width: 1
                            border.color: "#30363D"
                            radius: 6
                            
                            RowLayout {
                                anchors.fill: parent
                                anchors.margins: 8
                                spacing: 8
                                
                                Text {
                                    text: "⬇"
                                    font.pixelSize: 14
                                    color: selectedColor
                                }
                                
                                Text {
                                    Layout.fillWidth: true
                                    text: "Import Dataset"
                                    font.pixelSize: 12
                                    color: secondaryTextColor
                                }
                            }
                            
                            MouseArea {
                                id: importDataMouse
                                anchors.fill: parent
                                hoverEnabled: true
                                onClicked: console.log("Import dataset clicked")
                            }
                            
                            Behavior on color {
                                ColorAnimation { duration: animationDuration }
                            }
                        }
                    }
                }
                
                // Backtests Section  
                SidebarSection {
                    Layout.fillWidth: true
                    title: "Backtests"
                    expanded: true
                    
                    content: ColumnLayout {
                        spacing: 2
                        
                        SidebarItem {
                            Layout.fillWidth: true
                            text: "ICT Strategy Run #47"
                            subtitle: "ROI: +34.2% • Sharpe: 1.85"
                            icon: "✅"
                            iconColor: "#3FB950"
                            selected: true
                            onClicked: sideBar.backtestSelected("ict-47")
                        }
                        
                        SidebarItem {
                            Layout.fillWidth: true
                            text: "Ensemble Run #46"
                            subtitle: "ROI: +28.7% • Sharpe: 1.42"
                            icon: "✅"
                            iconColor: "#3FB950"
                            selected: false
                            onClicked: sideBar.backtestSelected("ensemble-46")
                        }
                        
                        SidebarItem {
                            Layout.fillWidth: true
                            text: "ML Strategy Run #45"
                            subtitle: "ROI: -5.3% • Sharpe: 0.23"
                            icon: "❌"
                            iconColor: "#F85149"
                            selected: false
                            onClicked: sideBar.backtestSelected("ml-45")
                        }
                        
                        SidebarItem {
                            Layout.fillWidth: true
                            text: "Momentum Run #44"
                            subtitle: "Running... 67%"
                            icon: "⏳"
                            iconColor: "#D29922"
                            selected: false
                            onClicked: sideBar.backtestSelected("momentum-44")
                        }
                        
                        // New backtest button
                        Rectangle {
                            Layout.fillWidth: true
                            Layout.preferredHeight: 40
                            Layout.margins: 8
                            color: selectedColor
                            radius: 8
                            
                            Text {
                                anchors.centerIn: parent
                                text: "New Backtest"
                                font.pixelSize: 13
                                font.bold: true
                                color: "white"
                            }
                            
                            MouseArea {
                                anchors.fill: parent
                                onClicked: console.log("New backtest clicked")
                            }
                        }
                    }
                }
                
                // Spacer to push everything to top
                Item {
                    Layout.fillHeight: true
                }
            }
        }
    }
}