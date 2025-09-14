import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Item {
    id: toastManager
    
    // Theme colors
    readonly property color cardColor: "#21262D"
    readonly property color borderColor: "#30363D"
    readonly property color textColor: "#F0F6FC"
    readonly property color successColor: "#3FB950"
    readonly property color warningColor: "#D29922"
    readonly property color errorColor: "#F85149"
    readonly property color infoColor: "#238636"
    
    function showMessage(message, type = "info", timeout = 3000) {
        toastModel.append({
            id: Date.now(),
            message: message,
            type: type,
            timeout: timeout
        })
    }
    
    function showSuccess(message, timeout = 3000) {
        showMessage(message, "success", timeout)
    }
    
    function showWarning(message, timeout = 4000) {
        showMessage(message, "warning", timeout)
    }
    
    function showError(message, timeout = 5000) {
        showMessage(message, "error", timeout)
    }
    
    function showInfo(message, timeout = 3000) {
        showMessage(message, "info", timeout)
    }
    
    Column {
        anchors.fill: parent
        spacing: 8
        
        Repeater {
            model: ListModel {
                id: toastModel
            }
            
            delegate: Rectangle {
                id: toastItem
                width: 300
                height: toastContent.implicitHeight + 24
                radius: 8
                color: cardColor
                border.width: 1
                border.color: borderColor
                
                // Left accent bar
                Rectangle {
                    anchors.left: parent.left
                    anchors.top: parent.top
                    anchors.bottom: parent.bottom
                    width: 4
                    radius: 2
                    color: {
                        switch(model.type) {
                            case "success": return successColor
                            case "warning": return warningColor
                            case "error": return errorColor
                            case "info": 
                            default: return infoColor
                        }
                    }
                }
                
                RowLayout {
                    id: toastContent
                    anchors.fill: parent
                    anchors.margins: 12
                    anchors.leftMargin: 20 // Account for accent bar
                    spacing: 12
                    
                    // Icon
                    Text {
                        text: {
                            switch(model.type) {
                                case "success": return "✅"
                                case "warning": return "⚠️"
                                case "error": return "❌"
                                case "info":
                                default: return "ℹ️"
                            }
                        }
                        font.pixelSize: 16
                    }
                    
                    // Message text
                    Text {
                        Layout.fillWidth: true
                        text: model.message
                        font.pixelSize: 13
                        color: textColor
                        wrapMode: Text.WordWrap
                    }
                    
                    // Close button
                    Rectangle {
                        Layout.preferredWidth: 20
                        Layout.preferredHeight: 20
                        radius: 10
                        color: closeMouseArea.containsMouse ? "#30363D" : "transparent"
                        
                        Text {
                            anchors.centerIn: parent
                            text: "×"
                            font.pixelSize: 14
                            color: textColor
                        }
                        
                        MouseArea {
                            id: closeMouseArea
                            anchors.fill: parent
                            hoverEnabled: true
                            onClicked: {
                                toastModel.remove(index)
                            }
                        }
                    }
                }
                
                // Auto-dismiss timer
                Timer {
                    interval: model.timeout
                    running: true
                    onTriggered: {
                        // Animate out then remove
                        fadeOutAnimation.start()
                    }
                }
                
                // Fade out animation
                NumberAnimation {
                    id: fadeOutAnimation
                    target: toastItem
                    property: "opacity"
                    from: 1.0
                    to: 0.0
                    duration: 300
                    easing.type: Easing.OutQuad
                    onFinished: {
                        toastModel.remove(index)
                    }
                }
                
                // Slide in animation when created
                PropertyAnimation {
                    id: slideInAnimation
                    target: toastItem
                    property: "x"
                    from: toastItem.width
                    to: 0
                    duration: 300
                    easing.type: Easing.OutBack
                    running: true
                }
                
                // Hover to pause auto-dismiss
                MouseArea {
                    anchors.fill: parent
                    hoverEnabled: true
                    acceptedButtons: Qt.NoButton
                    
                    onContainsMouseChanged: {
                        // Could pause/resume timer on hover if needed
                    }
                }
            }
        }
    }
}