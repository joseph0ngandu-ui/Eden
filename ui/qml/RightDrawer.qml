import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Rectangle {
    id: rightDrawer
    
    property bool collapsed: false
    property var gpuManager: null
    property var backtestManager: null
    
    // Theme colors
    readonly property color backgroundColor: "#0D1117"
    readonly property color cardColor: "#21262D"
    readonly property color borderColor: "#30363D"
    readonly property color textColor: "#F0F6FC"
    readonly property color secondaryTextColor: "#8B949E"
    readonly property color accentGreen: "#238636"
    readonly property color successColor: "#3FB950"
    readonly property color warningColor: "#D29922"
    readonly property color dangerColor: "#F85149"
    readonly property int animationDuration: 250
    
    color: backgroundColor
    border.width: 1
    border.color: borderColor
    
    // Collapse/Expand toggle button
    Rectangle {
        id: toggleButton
        anchors.left: parent.left
        anchors.verticalCenter: parent.verticalCenter
        anchors.leftMargin: -15
        width: 30
        height: 60
        radius: 15
        color: cardColor
        border.width: 1
        border.color: borderColor
        
        Text {
            anchors.centerIn: parent
            text: collapsed ? "◀" : "▶"
            font.pixelSize: 12
            color: secondaryTextColor
            
            Behavior on text {
                PropertyAnimation { duration: animationDuration }
            }
        }
        
        MouseArea {
            anchors.fill: parent
            onClicked: collapsed = !collapsed
        }
    }
    
    ColumnLayout {
        anchors.fill: parent
        anchors.margins: collapsed ? 8 : 16
        spacing: collapsed ? 4 : 12
        
        visible: !collapsed
        opacity: collapsed ? 0 : 1
        
        Behavior on opacity {
            NumberAnimation {
                duration: animationDuration
                easing.type: Easing.OutCubic
            }
        }
        
        // Header
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 40
            color: "transparent"
            
            RowLayout {
                anchors.fill: parent
                spacing: 8
                
                Text {
                    Layout.fillWidth: true
                    text: "Controls"
                    font.pixelSize: 16
                    font.bold: true
                    color: textColor
                }
                
                // Optional header actions
                Rectangle {
                    Layout.preferredWidth: 24
                    Layout.preferredHeight: 24
                    radius: 4
                    color: "transparent"
                    
                    Text {
                        anchors.centerIn: parent
                        text: "⚙"
                        font.pixelSize: 12
                        color: secondaryTextColor
                    }
                }
            }
        }
        
        // Main content area
        ScrollView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            
            contentWidth: availableWidth
            ScrollBar.horizontal.policy: ScrollBar.AlwaysOff
            ScrollBar.vertical.policy: ScrollBar.AsNeeded
            
            ColumnLayout {
                width: parent.width
                spacing: 16
                
                // Strategy Parameters Card
                ParametersCard {
                    Layout.fillWidth: true
                    title: "Strategy Parameters"
                    expandable: true
                    expanded: true
                    
                    content: ColumnLayout {
                        spacing: 12
                        
                        // Strategy selector
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 8
                            
                            Text {
                                text: "Strategy:"
                                font.pixelSize: 12
                                color: secondaryTextColor
                                Layout.preferredWidth: 60
                            }
                            
                            ComboBox {
                                id: strategyCombo
                                Layout.fillWidth: true
                                model: ["ICT", "Mean Reversion", "Momentum", "Ensemble", "ML Generated"]
                                currentIndex: 0
                                
                                background: Rectangle {
                                    color: "#30363D"
                                    border.color: "#484F58"
                                    border.width: 1
                                    radius: 6
                                }
                                
                                contentItem: Text {
                                    text: strategyCombo.displayText
                                    font.pixelSize: 12
                                    color: textColor
                                    verticalAlignment: Text.AlignVCenter
                                    leftPadding: 12
                                }
                            }
                        }
                        
                        // Symbol selector
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 8
                            
                            Text {
                                text: "Symbol:"
                                font.pixelSize: 12
                                color: secondaryTextColor
                                Layout.preferredWidth: 60
                            }
                            
                            ComboBox {
                                id: symbolCombo
                                Layout.fillWidth: true
                                model: ["XAUUSD", "EURUSD", "GBPUSD", "US30", "NAS100"]
                                currentIndex: 0
                                
                                background: Rectangle {
                                    color: "#30363D"
                                    border.color: "#484F58"
                                    border.width: 1
                                    radius: 6
                                }
                                
                                contentItem: Text {
                                    text: symbolCombo.displayText
                                    font.pixelSize: 12
                                    color: textColor
                                    verticalAlignment: Text.AlignVCenter
                                    leftPadding: 12
                                }
                            }
                        }
                        
                        // Starting cash
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 8
                            
                            Text {
                                text: "Capital:"
                                font.pixelSize: 12
                                color: secondaryTextColor
                                Layout.preferredWidth: 60
                            }
                            
                            SpinBox {
                                id: capitalSpinBox
                                Layout.fillWidth: true
                                from: 1000
                                to: 1000000
                                stepSize: 1000
                                value: 100000
                                
                                background: Rectangle {
                                    color: "#30363D"
                                    border.color: "#484F58"
                                    border.width: 1
                                    radius: 6
                                }
                                
                                contentItem: TextInput {
                                    text: parent.textFromValue(parent.value, parent.locale)
                                    font.pixelSize: 12
                                    color: textColor
                                    horizontalAlignment: Qt.AlignHCenter
                                    verticalAlignment: Qt.AlignVCenter
                                    readOnly: !parent.editable
                                    validator: parent.validator
                                    inputMethodHints: parent.inputMethodHints
                                }
                            }
                        }
                        
                        // Risk per trade slider
                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 4
                            
                            RowLayout {
                                Layout.fillWidth: true
                                
                                Text {
                                    text: "Risk per Trade:"
                                    font.pixelSize: 12
                                    color: secondaryTextColor
                                }
                                
                                Text {
                                    text: riskSlider.value.toFixed(1) + "%"
                                    font.pixelSize: 12
                                    font.bold: true
                                    color: textColor
                                }
                            }
                            
                            Slider {
                                id: riskSlider
                                Layout.fillWidth: true
                                from: 0.1
                                to: 5.0
                                stepSize: 0.1
                                value: 1.0
                                
                                background: Rectangle {
                                    x: riskSlider.leftPadding
                                    y: riskSlider.topPadding + riskSlider.availableHeight / 2 - height / 2
                                    implicitWidth: 200
                                    implicitHeight: 4
                                    width: riskSlider.availableWidth
                                    height: implicitHeight
                                    radius: 2
                                    color: "#30363D"
                                    
                                    Rectangle {
                                        width: riskSlider.visualPosition * parent.width
                                        height: parent.height
                                        color: accentGreen
                                        radius: 2
                                    }
                                }
                                
                                handle: Rectangle {
                                    x: riskSlider.leftPadding + riskSlider.visualPosition * (riskSlider.availableWidth - width)
                                    y: riskSlider.topPadding + riskSlider.availableHeight / 2 - height / 2
                                    implicitWidth: 20
                                    implicitHeight: 20
                                    radius: 10
                                    color: riskSlider.pressed ? "#2EA043" : accentGreen
                                    border.color: "#FFFFFF"
                                    border.width: 1
                                }
                            }
                        }
                        
                        // Commission and slippage
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 12
                            
                            ColumnLayout {
                                Layout.fillWidth: true
                                spacing: 4
                                
                                Text {
                                    text: "Commission (bps)"
                                    font.pixelSize: 11
                                    color: secondaryTextColor
                                }
                                
                                SpinBox {
                                    Layout.fillWidth: true
                                    from: 0
                                    to: 100
                                    stepSize: 1
                                    value: 1
                                    
                                    background: Rectangle {
                                        color: "#30363D"
                                        border.color: "#484F58"
                                        border.width: 1
                                        radius: 4
                                    }
                                }
                            }
                            
                            ColumnLayout {
                                Layout.fillWidth: true
                                spacing: 4
                                
                                Text {
                                    text: "Slippage (bps)"
                                    font.pixelSize: 11
                                    color: secondaryTextColor
                                }
                                
                                SpinBox {
                                    Layout.fillWidth: true
                                    from: 0
                                    to: 100
                                    stepSize: 1
                                    value: 1
                                    
                                    background: Rectangle {
                                        color: "#30363D"
                                        border.color: "#484F58"
                                        border.width: 1
                                        radius: 4
                                    }
                                }
                            }
                        }
                    }
                }
                
                // GPU Acceleration Card
                ParametersCard {
                    Layout.fillWidth: true
                    title: "GPU Acceleration"
                    expandable: true
                    expanded: true
                    
                    content: ColumnLayout {
                        spacing: 12
                        
                        // GPU Status
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 8
                            
                            Rectangle {
                                width: 8
                                height: 8
                                radius: 4
                                color: gpuManager?.isEnabled ? successColor : warningColor
                            }
                            
                            Text {
                                Layout.fillWidth: true
                                text: gpuManager?.statusText || "GPU Status Unknown"
                                font.pixelSize: 12
                                color: textColor
                            }
                        }
                        
                        // GPU Provider
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 8
                            
                            Text {
                                text: "Provider:"
                                font.pixelSize: 12
                                color: secondaryTextColor
                                Layout.preferredWidth: 60
                            }
                            
                            ComboBox {
                                Layout.fillWidth: true
                                model: ["Auto", "DirectML", "CUDA", "CPU Only"]
                                currentIndex: 0
                                
                                background: Rectangle {
                                    color: "#30363D"
                                    border.color: "#484F58"
                                    border.width: 1
                                    radius: 6
                                }
                                
                                contentItem: Text {
                                    text: parent.displayText
                                    font.pixelSize: 12
                                    color: textColor
                                    verticalAlignment: Text.AlignVCenter
                                    leftPadding: 12
                                }
                            }
                        }
                        
                        // Memory usage
                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 4
                            
                            RowLayout {
                                Layout.fillWidth: true
                                
                                Text {
                                    text: "VRAM Usage:"
                                    font.pixelSize: 12
                                    color: secondaryTextColor
                                }
                                
                                Text {
                                    text: (gpuManager?.memoryUsed || 0) + " MB / " + (gpuManager?.memoryTotal || 0) + " MB"
                                    font.pixelSize: 11
                                    color: textColor
                                }
                            }
                            
                            Rectangle {
                                Layout.fillWidth: true
                                height: 6
                                radius: 3
                                color: "#30363D"
                                
                                Rectangle {
                                    width: parent.width * Math.min(1.0, (gpuManager?.memoryUsed || 0) / Math.max(1, gpuManager?.memoryTotal || 1))
                                    height: parent.height
                                    radius: 3
                                    color: {
                                        let usage = (gpuManager?.memoryUsed || 0) / Math.max(1, gpuManager?.memoryTotal || 1)
                                        if (usage > 0.9) return dangerColor
                                        if (usage > 0.7) return warningColor
                                        return successColor
                                    }
                                }
                            }
                        }
                    }
                }
                
                // Run Controls Card
                ParametersCard {
                    Layout.fillWidth: true
                    title: "Run Controls"
                    expandable: false
                    expanded: true
                    
                    content: ColumnLayout {
                        spacing: 12
                        
                        // Main run button
                        Rectangle {
                            Layout.fillWidth: true
                            Layout.preferredHeight: 44
                            radius: 8
                            color: runMouseArea.pressed ? "#2EA043" : (runMouseArea.containsMouse ? "#2EA043" : accentGreen)
                            
                            Text {
                                anchors.centerIn: parent
                                text: "Run Backtest"
                                font.pixelSize: 14
                                font.bold: true
                                color: "white"
                            }
                            
                            MouseArea {
                                id: runMouseArea
                                anchors.fill: parent
                                hoverEnabled: true
                                onClicked: {
                                    console.log("Run backtest clicked")
                                    // TODO: Trigger backtest via backtestManager
                                }
                            }
                            
                            Behavior on color {
                                ColorAnimation { duration: 150 }
                            }
                        }
                        
                        // Secondary actions
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 8
                            
                            Rectangle {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 36
                                radius: 6
                                color: optimizeMouseArea.containsMouse ? "#30363D" : "transparent"
                                border.width: 1
                                border.color: "#484F58"
                                
                                Text {
                                    anchors.centerIn: parent
                                    text: "Optimize"
                                    font.pixelSize: 12
                                    color: textColor
                                }
                                
                                MouseArea {
                                    id: optimizeMouseArea
                                    anchors.fill: parent
                                    hoverEnabled: true
                                    onClicked: console.log("Optimize clicked")
                                }
                            }
                            
                            Rectangle {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 36
                                radius: 6
                                color: stopMouseArea.containsMouse ? "#30363D" : "transparent"
                                border.width: 1
                                border.color: "#484F58"
                                
                                Text {
                                    anchors.centerIn: parent
                                    text: "Stop"
                                    font.pixelSize: 12
                                    color: dangerColor
                                }
                                
                                MouseArea {
                                    id: stopMouseArea
                                    anchors.fill: parent
                                    hoverEnabled: true
                                    onClicked: console.log("Stop clicked")
                                }
                            }
                        }
                        
                        // Progress indicator
                        Rectangle {
                            Layout.fillWidth: true
                            Layout.preferredHeight: 8
                            radius: 4
                            color: "#30363D"
                            visible: backtestManager?.isRunning || false
                            
                            Rectangle {
                                width: parent.width * Math.max(0, Math.min(1, backtestManager?.progress || 0))
                                height: parent.height
                                radius: 4
                                color: accentGreen
                                
                                Behavior on width {
                                    NumberAnimation { duration: 200 }
                                }
                            }
                        }
                        
                        // Status text
                        Text {
                            Layout.fillWidth: true
                            text: backtestManager?.statusMessage || "Ready"
                            font.pixelSize: 11
                            color: secondaryTextColor
                            horizontalAlignment: Text.AlignHCenter
                            wrapMode: Text.WordWrap
                        }
                    }
                }
                
                // Spacer to push content to top
                Item {
                    Layout.fillHeight: true
                }
            }
        }
    }
    
    // Collapsed state content (icon buttons)
    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 8
        spacing: 8
        
        visible: collapsed
        opacity: collapsed ? 1 : 0
        
        Behavior on opacity {
            NumberAnimation {
                duration: animationDuration
                easing.type: Easing.OutCubic
            }
        }
        
        // Collapsed icon buttons
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 44
            radius: 8
            color: collapsedRunMouseArea.containsMouse ? "#2EA043" : accentGreen
            
            Text {
                anchors.centerIn: parent
                text: "▶"
                font.pixelSize: 18
                color: "white"
            }
            
            MouseArea {
                id: collapsedRunMouseArea
                anchors.fill: parent
                hoverEnabled: true
                onClicked: {
                    collapsed = false
                    // Run backtest
                }
            }
        }
        
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 36
            radius: 6
            color: collapsedGpuMouseArea.containsMouse ? "#30363D" : "transparent"
            border.width: 1
            border.color: "#484F58"
            
            Text {
                anchors.centerIn: parent
                text: "🖥"
                font.pixelSize: 16
            }
            
            MouseArea {
                id: collapsedGpuMouseArea
                anchors.fill: parent
                hoverEnabled: true
                onClicked: collapsed = false
            }
        }
        
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 36
            radius: 6
            color: collapsedSettingsMouseArea.containsMouse ? "#30363D" : "transparent"
            border.width: 1
            border.color: "#484F58"
            
            Text {
                anchors.centerIn: parent
                text: "⚙"
                font.pixelSize: 16
                color: secondaryTextColor
            }
            
            MouseArea {
                id: collapsedSettingsMouseArea
                anchors.fill: parent
                hoverEnabled: true
                onClicked: collapsed = false
            }
        }
        
        Item { Layout.fillHeight: true }
    }
}