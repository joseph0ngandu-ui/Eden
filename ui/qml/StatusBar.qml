import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Rectangle {
    id: statusBar
    
    property string statusText: "Ready"
    property string version: "1.0.0"
    property bool isInitialized: false
    
    // Theme colors
    readonly property color backgroundColor: "#161B22"
    readonly property color borderColor: "#30363D"
    readonly property color textColor: "#F0F6FC"
    readonly property color secondaryTextColor: "#8B949E"
    readonly property color accentGreen: "#238636"
    readonly property color warningColor: "#D29922"
    
    color: backgroundColor
    border.width: 1
    border.color: borderColor
    
    RowLayout {
        anchors.fill: parent
        anchors.margins: 8
        spacing: 16
        
        // Status indicator
        Rectangle {
            width: 8
            height: 8
            radius: 4
            color: isInitialized ? accentGreen : warningColor
            
            // Subtle pulsing when not initialized
            SequentialAnimation {
                running: !isInitialized
                loops: Animation.Infinite
                
                PropertyAnimation {
                    target: parent
                    property: "opacity"
                    from: 1.0
                    to: 0.3
                    duration: 1000
                }
                
                PropertyAnimation {
                    target: parent
                    property: "opacity"
                    from: 0.3
                    to: 1.0
                    duration: 1000
                }
            }
        }
        
        // Status text
        Text {
            text: statusText
            font.pixelSize: 11
            color: textColor
        }
        
        // Separator
        Rectangle {
            width: 1
            height: 12
            color: borderColor
        }
        
        // Memory usage (placeholder)
        Text {
            text: "RAM: 512MB"
            font.pixelSize: 11
            color: secondaryTextColor
        }
        
        // Separator
        Rectangle {
            width: 1
            height: 12
            color: borderColor
        }
        
        // Connection status
        RowLayout {
            spacing: 4
            
            Rectangle {
                width: 6
                height: 6
                radius: 3
                color: accentGreen // Connected
            }
            
            Text {
                text: "Workers Connected"
                font.pixelSize: 11
                color: secondaryTextColor
            }
        }
        
        // Spacer
        Item {
            Layout.fillWidth: true
        }
        
        // Current time
        Text {
            id: timeText
            text: new Date().toLocaleTimeString()
            font.pixelSize: 11
            color: secondaryTextColor
            
            Timer {
                interval: 1000
                running: true
                repeat: true
                onTriggered: timeText.text = new Date().toLocaleTimeString()
            }
        }
        
        // Separator
        Rectangle {
            width: 1
            height: 12
            color: borderColor
        }
        
        // Version
        Text {
            text: "v" + version
            font.pixelSize: 11
            color: secondaryTextColor
        }
    }
}