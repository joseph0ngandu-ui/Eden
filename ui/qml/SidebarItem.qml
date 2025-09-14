import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Rectangle {
    id: item
    
    property string text: ""
    property string subtitle: ""
    property string icon: ""
    property color iconColor: "#8B949E"
    property bool selected: false
    
    signal clicked()
    
    // Theme colors
    readonly property color backgroundColor: "#0D1117"
    readonly property color hoverColor: "#21262D"
    readonly property color selectedColor: "#238636"
    readonly property color selectedBackgroundColor: "#1a472a"
    readonly property color textColor: "#F0F6FC"
    readonly property color secondaryTextColor: "#8B949E"
    readonly property int animationDuration: 150
    
    Layout.preferredHeight: 50
    color: {
        if (selected) return selectedBackgroundColor
        if (mouseArea.containsMouse) return hoverColor
        return backgroundColor
    }
    radius: 6
    
    // Left border accent for selected items
    Rectangle {
        anchors.left: parent.left
        anchors.top: parent.top
        anchors.bottom: parent.bottom
        width: 3
        color: selectedColor
        radius: 1.5
        visible: selected
        
        Behavior on visible {
            NumberAnimation { duration: animationDuration }
        }
    }
    
    RowLayout {
        anchors.fill: parent
        anchors.margins: 12
        anchors.leftMargin: selected ? 16 : 12  // Extra margin when selected
        spacing: 12
        
        // Icon
        Rectangle {
            Layout.preferredWidth: 24
            Layout.preferredHeight: 24
            color: "transparent"
            
            Text {
                anchors.centerIn: parent
                text: item.icon
                font.pixelSize: 16
                color: item.iconColor
            }
        }
        
        // Text content
        ColumnLayout {
            Layout.fillWidth: true
            spacing: 2
            
            // Main text
            Text {
                Layout.fillWidth: true
                text: item.text
                font.pixelSize: 13
                font.weight: selected ? Font.DemiBold : Font.Normal
                color: selected ? "#FFFFFF" : textColor
                elide: Text.ElideRight
                
                Behavior on color {
                    ColorAnimation { duration: animationDuration }
                }
                
                Behavior on font.weight {
                    NumberAnimation { duration: animationDuration }
                }
            }
            
            // Subtitle
            Text {
                Layout.fillWidth: true
                text: item.subtitle
                font.pixelSize: 11
                color: selected ? "#E6EDF3" : secondaryTextColor
                elide: Text.ElideRight
                visible: item.subtitle.length > 0
                
                Behavior on color {
                    ColorAnimation { duration: animationDuration }
                }
            }
        }
        
        // Optional status indicator or action
        Rectangle {
            Layout.preferredWidth: 6
            Layout.preferredHeight: 6
            radius: 3
            color: selected ? selectedColor : "transparent"
            
            Behavior on color {
                ColorAnimation { duration: animationDuration }
            }
        }
    }
    
    MouseArea {
        id: mouseArea
        anchors.fill: parent
        hoverEnabled: true
        onClicked: item.clicked()
        
        // Ripple effect on click
        Rectangle {
            id: ripple
            anchors.centerIn: parent
            width: 0
            height: 0
            radius: width / 2
            color: Qt.rgba(1, 1, 1, 0.1)
            
            states: State {
                name: "active"
                when: mouseArea.pressed
                PropertyChanges {
                    target: ripple
                    width: Math.max(item.width, item.height) * 1.2
                    height: width
                }
            }
            
            transitions: Transition {
                from: ""
                to: "active"
                NumberAnimation {
                    properties: "width,height"
                    duration: 150
                    easing.type: Easing.OutCubic
                }
            }
        }
    }
    
    Behavior on color {
        ColorAnimation { duration: animationDuration }
    }
    
    Behavior on anchors.leftMargin {
        NumberAnimation { duration: animationDuration }
    }
}