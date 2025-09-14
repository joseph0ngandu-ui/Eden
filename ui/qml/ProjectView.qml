import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Rectangle {
    id: projectView
    
    // Theme colors
    readonly property color backgroundColor: "#0D1117"
    readonly property color textColor: "#F0F6FC"
    readonly property color secondaryTextColor: "#8B949E"
    
    color: backgroundColor
    
    function loadProject(projectId) {
        console.log("Loading project:", projectId)
        // Implementation for loading project data
    }
    
    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 32
        
        Text {
            text: "Project View"
            font.pixelSize: 24
            font.bold: true
            color: textColor
        }
        
        Text {
            text: "Project management interface will be implemented here"
            font.pixelSize: 14
            color: secondaryTextColor
        }
        
        Item { Layout.fillHeight: true }
    }
}