import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Rectangle {
    id: card
    
    property string title: ""
    property Component content: null
    property bool expandable: false
    property bool expanded: true
    
    // Theme colors
    readonly property color cardColor: "#21262D"
    readonly property color borderColor: "#30363D"
    readonly property color textColor: "#F0F6FC"
    readonly property color secondaryTextColor: "#8B949E"
    readonly property color hoverColor: "#30363D"
    readonly property int animationDuration: 200
    
    color: cardColor
    border.width: 1
    border.color: borderColor
    radius: 8
    
    Layout.preferredHeight: headerHeight + (expanded ? contentHeight : 0) + 32
    
    readonly property int headerHeight: 40
    readonly property int contentHeight: contentLoader.item ? contentLoader.item.implicitHeight : 0
    
    Behavior on Layout.preferredHeight {
        NumberAnimation {
            duration: animationDuration
            easing.type: Easing.OutCubic
        }
    }
    
    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 16
        spacing: 12
        
        // Card Header
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: headerHeight
            color: expandable && headerMouseArea.containsMouse ? hoverColor : "transparent"
            radius: 6
            
            RowLayout {
                anchors.fill: parent
                anchors.margins: expandable ? 8 : 0
                spacing: 8
                
                // Expand/Collapse indicator (only if expandable)
                Rectangle {
                    Layout.preferredWidth: 16
                    Layout.preferredHeight: 16
                    color: "transparent"
                    visible: expandable
                    
                    Text {
                        anchors.centerIn: parent
                        text: expanded ? "▼" : "▶"
                        font.pixelSize: 10
                        color: secondaryTextColor
                        
                        Behavior on rotation {
                            NumberAnimation {
                                duration: animationDuration
                                easing.type: Easing.OutCubic
                            }
                        }
                    }
                }
                
                // Card title
                Text {
                    Layout.fillWidth: true
                    text: card.title
                    font.pixelSize: 14
                    font.bold: true
                    color: textColor
                }
                
                // Optional header actions
                Item {
                    Layout.preferredWidth: 16
                    Layout.preferredHeight: 16
                    visible: false  // Can be used for header actions
                }
            }
            
            MouseArea {
                id: headerMouseArea
                anchors.fill: parent
                hoverEnabled: expandable
                enabled: expandable
                onClicked: card.expanded = !card.expanded
            }
            
            Behavior on color {
                ColorAnimation { duration: animationDuration }
            }
        }
        
        // Card Content
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: expanded ? contentHeight : 0
            color: "transparent"
            clip: true
            
            Loader {
                id: contentLoader
                anchors.fill: parent
                sourceComponent: card.content
                active: expanded
            }
            
            Behavior on Layout.preferredHeight {
                NumberAnimation {
                    duration: animationDuration
                    easing.type: Easing.OutCubic
                }
            }
        }
    }
}