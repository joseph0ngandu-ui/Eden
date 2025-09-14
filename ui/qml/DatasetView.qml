import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Rectangle {
    id: datasetView
    
    // Theme colors
    readonly property color backgroundColor: "#0D1117"
    readonly property color textColor: "#F0F6FC"
    readonly property color secondaryTextColor: "#8B949E"
    
    color: backgroundColor
    
    function loadDataset(datasetId) {
        console.log("Loading dataset:", datasetId)
        // Implementation for loading dataset data
    }
    
    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 32
        
        Text {
            text: "Dataset View"
            font.pixelSize: 24
            font.bold: true
            color: textColor
        }
        
        Text {
            text: "Dataset management and preview interface will be implemented here"
            font.pixelSize: 14
            color: secondaryTextColor
        }
        
        Item { Layout.fillHeight: true }
    }
}