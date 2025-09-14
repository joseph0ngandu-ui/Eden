import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtGraphicalEffects 1.15

Rectangle {
    id: loadingOverlay
    
    property string statusMessage: "Loading..."
    
    // Theme colors
    readonly property color backgroundColor: "#0D1117"
    readonly property color cardColor: "#21262D"
    readonly property color textColor: "#F0F6FC"
    readonly property color secondaryTextColor: "#8B949E"
    readonly property color accentGreen: "#238636"
    
    color: Qt.rgba(0.05, 0.067, 0.09, 0.95) // Semi-transparent dark overlay
    
    // Blur background
    FastBlur {
        anchors.fill: parent
        source: parent.parent
        radius: 32
        visible: false // Disable for performance, could be enabled for premium effect
    }
    
    Rectangle {
        anchors.centerIn: parent
        width: 300
        height: 200
        radius: 16
        color: cardColor
        border.width: 1
        border.color: "#30363D"
        
        // Subtle shadow effect
        Rectangle {
            anchors.fill: parent
            anchors.margins: -4
            radius: parent.radius + 4
            color: Qt.rgba(0, 0, 0, 0.3)
            z: parent.z - 1
        }
        
        ColumnLayout {
            anchors.fill: parent
            anchors.margins: 32
            spacing: 24
            
            // Eden logo/icon
            Rectangle {
                Layout.preferredWidth: 64
                Layout.preferredHeight: 64
                Layout.alignment: Qt.AlignHCenter
                radius: 16
                color: accentGreen
                
                Text {
                    anchors.centerIn: parent
                    text: "E"
                    font.pixelSize: 32
                    font.bold: true
                    color: "white"
                }
                
                // Subtle pulsing animation
                SequentialAnimation {
                    running: true
                    loops: Animation.Infinite
                    
                    PropertyAnimation {
                        target: parent
                        property: "scale"
                        from: 1.0
                        to: 1.1
                        duration: 1500
                        easing.type: Easing.InOutQuad
                    }
                    
                    PropertyAnimation {
                        target: parent
                        property: "scale"
                        from: 1.1
                        to: 1.0
                        duration: 1500
                        easing.type: Easing.InOutQuad
                    }
                }
            }
            
            // Loading text
            Text {
                Layout.fillWidth: true
                text: "Eden Trading System"
                font.pixelSize: 20
                font.bold: true
                color: textColor
                horizontalAlignment: Text.AlignHCenter
            }
            
            // Status message
            Text {
                Layout.fillWidth: true
                text: statusMessage
                font.pixelSize: 14
                color: secondaryTextColor
                horizontalAlignment: Text.AlignHCenter
                wrapMode: Text.WordWrap
            }
            
            // Progress indicator
            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 4
                radius: 2
                color: "#30363D"
                
                Rectangle {
                    id: progressBar
                    height: parent.height
                    radius: 2
                    color: accentGreen
                    
                    // Animated progress bar
                    SequentialAnimation {
                        running: true
                        loops: Animation.Infinite
                        
                        PropertyAnimation {
                            target: progressBar
                            property: "width"
                            from: 0
                            to: progressBar.parent.width
                            duration: 2000
                            easing.type: Easing.InOutQuad
                        }
                        
                        PauseAnimation {
                            duration: 500
                        }
                        
                        PropertyAnimation {
                            target: progressBar
                            property: "width"
                            from: progressBar.parent.width
                            to: 0
                            duration: 2000
                            easing.type: Easing.InOutQuad
                        }
                        
                        PauseAnimation {
                            duration: 500
                        }
                    }
                }
            }
            
            // Version info
            Text {
                Layout.fillWidth: true
                text: "v1.0.0"
                font.pixelSize: 11
                color: Qt.rgba(secondaryTextColor.r, secondaryTextColor.g, secondaryTextColor.b, 0.6)
                horizontalAlignment: Text.AlignHCenter
            }
        }
    }
    
    // Click to potentially skip (if implemented)
    MouseArea {
        anchors.fill: parent
        onClicked: {
            // Could implement skip loading if desired
            console.log("Loading overlay clicked")
        }
    }
}