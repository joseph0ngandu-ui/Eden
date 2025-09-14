import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Rectangle {
    id: section
    
    property string title: ""
    property bool expanded: true
    property Component content: null
    
    readonly property color backgroundColor: "#0D1117"
    readonly property color headerColor: "#161B22"
    readonly property color textColor: "#F0F6FC"
    readonly property color secondaryTextColor: "#8B949E"
    readonly property color hoverColor: "#21262D"
    readonly property int animationDuration: 200
    
    color: backgroundColor
    Layout.preferredHeight: headerHeight + (expanded ? contentHeight : 0)
    
    readonly property int headerHeight: 40
    readonly property int contentHeight: contentLoader.item ? contentLoader.item.implicitHeight + 16 : 0
    
    Behavior on Layout.preferredHeight {
        NumberAnimation {
            duration: animationDuration
            easing.type: Easing.OutCubic
        }
    }
    
    ColumnLayout {
        anchors.fill: parent
        spacing: 0
        
        // Section Header
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: headerHeight
            color: headerMouseArea.containsMouse ? hoverColor : headerColor
            
            RowLayout {
                anchors.fill: parent
                anchors.margins: 12
                spacing: 8
                
                // Expand/Collapse indicator
                Rectangle {
                    Layout.preferredWidth: 16
                    Layout.preferredHeight: 16
                    color: "transparent"
                    
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
                
                // Section title
                Text {
                    Layout.fillWidth: true
                    text: section.title
                    font.pixelSize: 13
                    font.bold: true
                    color: textColor
                }
                
                // Optional section action (count, etc)
                Text {
                    text: ""  // Can be overridden
                    font.pixelSize: 11
                    color: secondaryTextColor
                }
            }
            
            MouseArea {
                id: headerMouseArea
                anchors.fill: parent
                hoverEnabled: true
                onClicked: section.expanded = !section.expanded
            }
            
            Behavior on color {
                ColorAnimation { duration: animationDuration }
            }
        }
        
        // Section Content
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: expanded ? contentHeight : 0
            color: backgroundColor
            clip: true
            
            Loader {
                id: contentLoader
                anchors.fill: parent
                anchors.margins: 8
                sourceComponent: section.content
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