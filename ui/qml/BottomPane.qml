import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtCharts 2.15

Rectangle {
    id: bottomPane
    
    property bool collapsed: false
    property var backtestManager: null
    
    // Theme colors
    readonly property color backgroundColor: "#0D1117"
    readonly property color surfaceColor: "#161B22"
    readonly property color cardColor: "#21262D"
    readonly property color borderColor: "#30363D"
    readonly property color textColor: "#F0F6FC"
    readonly property color secondaryTextColor: "#8B949E"
    readonly property color accentGreen: "#238636"
    readonly property color successColor: "#3FB950"
    readonly property color dangerColor: "#F85149"
    readonly property color warningColor: "#D29922"
    readonly property int animationDuration: 250
    
    color: backgroundColor
    border.width: 1
    border.color: borderColor
    
    // Collapse/Expand toggle button
    Rectangle {
        id: toggleButton
        anchors.top: parent.top
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.topMargin: -15
        width: 60
        height: 30
        radius: 15
        color: cardColor
        border.width: 1
        border.color: borderColor
        
        Text {
            anchors.centerIn: parent
            text: collapsed ? "▲" : "▼"
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
        anchors.margins: 1
        spacing: 0
        
        visible: !collapsed
        opacity: collapsed ? 0 : 1
        
        Behavior on opacity {
            NumberAnimation {
                duration: animationDuration
                easing.type: Easing.OutCubic
            }
        }
        
        // Tab Bar
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 48
            color: surfaceColor
            border.width: 1
            border.color: borderColor
            
            TabBar {
                id: tabBar
                anchors.fill: parent
                anchors.margins: 8
                
                background: Rectangle {
                    color: "transparent"
                }
                
                TabButton {
                    text: "Logs"
                    width: Math.max(80, implicitWidth)
                    
                    background: Rectangle {
                        color: parent.checked ? cardColor : "transparent"
                        border.width: parent.checked ? 1 : 0
                        border.color: borderColor
                        radius: 6
                        
                        Behavior on color {
                            ColorAnimation { duration: 150 }
                        }
                    }
                    
                    contentItem: RowLayout {
                        spacing: 8
                        
                        Text {
                            text: "📋"
                            font.pixelSize: 14
                        }
                        
                        Text {
                            text: parent.parent.text
                            font.pixelSize: 13
                            color: parent.parent.checked ? textColor : secondaryTextColor
                            
                            Behavior on color {
                                ColorAnimation { duration: 150 }
                            }
                        }
                        
                        // Log count badge
                        Rectangle {
                            Layout.preferredWidth: 20
                            Layout.preferredHeight: 16
                            radius: 8
                            color: warningColor
                            visible: logModel.count > 0
                            
                            Text {
                                anchors.centerIn: parent
                                text: Math.min(99, logModel.count)
                                font.pixelSize: 10
                                font.bold: true
                                color: "white"
                            }
                        }
                    }
                }
                
                TabButton {
                    text: "Trades"
                    width: Math.max(80, implicitWidth)
                    
                    background: Rectangle {
                        color: parent.checked ? cardColor : "transparent"
                        border.width: parent.checked ? 1 : 0
                        border.color: borderColor
                        radius: 6
                        
                        Behavior on color {
                            ColorAnimation { duration: 150 }
                        }
                    }
                    
                    contentItem: RowLayout {
                        spacing: 8
                        
                        Text {
                            text: "💹"
                            font.pixelSize: 14
                        }
                        
                        Text {
                            text: parent.parent.text
                            font.pixelSize: 13
                            color: parent.parent.checked ? textColor : secondaryTextColor
                            
                            Behavior on color {
                                ColorAnimation { duration: 150 }
                            }
                        }
                        
                        // Trade count badge
                        Rectangle {
                            Layout.preferredWidth: 20
                            Layout.preferredHeight: 16
                            radius: 8
                            color: accentGreen
                            visible: tradesModel.count > 0
                            
                            Text {
                                anchors.centerIn: parent
                                text: Math.min(99, tradesModel.count)
                                font.pixelSize: 10
                                font.bold: true
                                color: "white"
                            }
                        }
                    }
                }
                
                TabButton {
                    text: "Equity"
                    width: Math.max(80, implicitWidth)
                    
                    background: Rectangle {
                        color: parent.checked ? cardColor : "transparent"
                        border.width: parent.checked ? 1 : 0
                        border.color: borderColor
                        radius: 6
                        
                        Behavior on color {
                            ColorAnimation { duration: 150 }
                        }
                    }
                    
                    contentItem: RowLayout {
                        spacing: 8
                        
                        Text {
                            text: "📈"
                            font.pixelSize: 14
                        }
                        
                        Text {
                            text: parent.parent.text
                            font.pixelSize: 13
                            color: parent.parent.checked ? textColor : secondaryTextColor
                            
                            Behavior on color {
                                ColorAnimation { duration: 150 }
                            }
                        }
                    }
                }
                
                // Performance summary in tab bar
                Item {
                    Layout.fillWidth: true
                }
                
                Rectangle {
                    Layout.preferredWidth: 200
                    Layout.preferredHeight: 32
                    color: cardColor
                    border.width: 1
                    border.color: borderColor
                    radius: 6
                    
                    RowLayout {
                        anchors.fill: parent
                        anchors.margins: 8
                        spacing: 12
                        
                        ColumnLayout {
                            spacing: 2
                            
                            Text {
                                text: "P&L"
                                font.pixelSize: 10
                                color: secondaryTextColor
                            }
                            
                            Text {
                                text: backtestManager?.totalPnl ? "$" + backtestManager.totalPnl.toFixed(2) : "$0.00"
                                font.pixelSize: 11
                                font.bold: true
                                color: (backtestManager?.totalPnl || 0) >= 0 ? successColor : dangerColor
                            }
                        }
                        
                        Rectangle {
                            width: 1
                            Layout.fillHeight: true
                            color: borderColor
                        }
                        
                        ColumnLayout {
                            spacing: 2
                            
                            Text {
                                text: "Trades"
                                font.pixelSize: 10
                                color: secondaryTextColor
                            }
                            
                            Text {
                                text: backtestManager?.totalTrades || "0"
                                font.pixelSize: 11
                                font.bold: true
                                color: textColor
                            }
                        }
                        
                        Rectangle {
                            width: 1
                            Layout.fillHeight: true
                            color: borderColor
                        }
                        
                        ColumnLayout {
                            spacing: 2
                            
                            Text {
                                text: "Win Rate"
                                font.pixelSize: 10
                                color: secondaryTextColor
                            }
                            
                            Text {
                                text: backtestManager?.winRate ? (backtestManager.winRate * 100).toFixed(1) + "%" : "0%"
                                font.pixelSize: 11
                                font.bold: true
                                color: textColor
                            }
                        }
                    }
                }
            }
        }
        
        // Tab Content
        StackLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            currentIndex: tabBar.currentIndex
            
            // Logs Tab
            Rectangle {
                color: backgroundColor
                
                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 12
                    spacing: 8
                    
                    // Log controls
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 8
                        
                        ComboBox {
                            id: logLevelFilter
                            model: ["All", "DEBUG", "INFO", "WARNING", "ERROR"]
                            currentIndex: 1  // INFO
                            
                            background: Rectangle {
                                color: "#30363D"
                                border.color: "#484F58"
                                border.width: 1
                                radius: 6
                            }
                            
                            contentItem: Text {
                                text: logLevelFilter.displayText
                                font.pixelSize: 12
                                color: textColor
                                verticalAlignment: Text.AlignVCenter
                                leftPadding: 12
                            }
                        }
                        
                        Item { Layout.fillWidth: true }
                        
                        Rectangle {
                            Layout.preferredWidth: 100
                            Layout.preferredHeight: 32
                            radius: 6
                            color: clearLogsMouseArea.containsMouse ? "#30363D" : "transparent"
                            border.width: 1
                            border.color: "#484F58"
                            
                            Text {
                                anchors.centerIn: parent
                                text: "Clear Logs"
                                font.pixelSize: 12
                                color: dangerColor
                            }
                            
                            MouseArea {
                                id: clearLogsMouseArea
                                anchors.fill: parent
                                hoverEnabled: true
                                onClicked: logModel.clear()
                            }
                        }
                    }
                    
                    // Log list
                    ScrollView {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        
                        ListView {
                            id: logListView
                            model: ListModel {
                                id: logModel
                                
                                Component.onCompleted: {
                                    // Sample log entries
                                    append({
                                        timestamp: new Date().toLocaleTimeString(),
                                        level: "INFO",
                                        message: "Eden Trading System started",
                                        category: "system"
                                    })
                                    append({
                                        timestamp: new Date().toLocaleTimeString(),
                                        level: "INFO",
                                        message: "Loading XAUUSD data for backtest",
                                        category: "data"
                                    })
                                    append({
                                        timestamp: new Date().toLocaleTimeString(),
                                        level: "DEBUG",
                                        message: "Strategy signals generated: 3 buy, 2 sell",
                                        category: "strategy"
                                    })
                                }
                            }
                            
                            delegate: Rectangle {
                                width: logListView.width
                                height: 28
                                color: index % 2 == 0 ? "transparent" : Qt.rgba(1, 1, 1, 0.02)
                                
                                RowLayout {
                                    anchors.fill: parent
                                    anchors.margins: 4
                                    spacing: 8
                                    
                                    Text {
                                        text: model.timestamp
                                        font.pixelSize: 11
                                        font.family: "Consolas, Monaco, monospace"
                                        color: secondaryTextColor
                                        Layout.preferredWidth: 80
                                    }
                                    
                                    Rectangle {
                                        Layout.preferredWidth: 50
                                        Layout.preferredHeight: 16
                                        radius: 3
                                        color: {
                                            switch(model.level) {
                                                case "DEBUG": return "#30363D"
                                                case "INFO": return accentGreen
                                                case "WARNING": return warningColor
                                                case "ERROR": return dangerColor
                                                default: return "#30363D"
                                            }
                                        }
                                        
                                        Text {
                                            anchors.centerIn: parent
                                            text: model.level
                                            font.pixelSize: 9
                                            font.bold: true
                                            color: "white"
                                        }
                                    }
                                    
                                    Text {
                                        Layout.fillWidth: true
                                        text: model.message
                                        font.pixelSize: 12
                                        font.family: "Consolas, Monaco, monospace"
                                        color: textColor
                                        elide: Text.ElideRight
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            // Trades Tab
            Rectangle {
                color: backgroundColor
                
                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 12
                    spacing: 8
                    
                    // Trade controls
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 8
                        
                        Text {
                            text: "Recent Trades"
                            font.pixelSize: 14
                            font.bold: true
                            color: textColor
                        }
                        
                        Item { Layout.fillWidth: true }
                        
                        Rectangle {
                            Layout.preferredWidth: 120
                            Layout.preferredHeight: 32
                            radius: 6
                            color: exportTradesMouseArea.containsMouse ? "#30363D" : "transparent"
                            border.width: 1
                            border.color: "#484F58"
                            
                            Text {
                                anchors.centerIn: parent
                                text: "Export CSV"
                                font.pixelSize: 12
                                color: textColor
                            }
                            
                            MouseArea {
                                id: exportTradesMouseArea
                                anchors.fill: parent
                                hoverEnabled: true
                                onClicked: console.log("Export trades clicked")
                            }
                        }
                    }
                    
                    // Trades table
                    ScrollView {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        
                        TableView {
                            id: tradesTable
                            
                            model: ListModel {
                                id: tradesModel
                                
                                Component.onCompleted: {
                                    // Sample trade data
                                    append({
                                        id: "1",
                                        timestamp: "2024-01-15 14:30:00",
                                        symbol: "XAUUSD",
                                        side: "BUY",
                                        quantity: 0.1,
                                        entryPrice: 2645.23,
                                        exitPrice: 2652.15,
                                        pnl: 69.20,
                                        commission: 2.65
                                    })
                                    append({
                                        id: "2",
                                        timestamp: "2024-01-15 15:45:00",
                                        symbol: "XAUUSD",
                                        side: "SELL",
                                        quantity: 0.1,
                                        entryPrice: 2650.80,
                                        exitPrice: 2645.10,
                                        pnl: 57.00,
                                        commission: 2.65
                                    })
                                }
                            }
                            
                            delegate: Rectangle {
                                width: 120
                                height: 32
                                color: row % 2 == 0 ? "transparent" : Qt.rgba(1, 1, 1, 0.02)
                                border.width: 1
                                border.color: "transparent"
                                
                                Text {
                                    anchors.centerIn: parent
                                    text: {
                                        switch(column) {
                                            case 0: return model ? model.timestamp : ""
                                            case 1: return model ? model.symbol : ""
                                            case 2: return model ? model.side : ""
                                            case 3: return model ? model.quantity : ""
                                            case 4: return model ? "$" + model.entryPrice.toFixed(2) : ""
                                            case 5: return model ? "$" + model.exitPrice.toFixed(2) : ""
                                            case 6: return model ? "$" + model.pnl.toFixed(2) : ""
                                            default: return ""
                                        }
                                    }
                                    font.pixelSize: 11
                                    font.family: "Consolas, Monaco, monospace"
                                    color: column === 6 ? (model && model.pnl >= 0 ? successColor : dangerColor) : textColor
                                }
                            }
                        }
                    }
                }
            }
            
            // Equity Tab
            Rectangle {
                color: backgroundColor
                
                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 12
                    spacing: 8
                    
                    // Equity chart controls
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 8
                        
                        Text {
                            text: "Equity Curve"
                            font.pixelSize: 14
                            font.bold: true
                            color: textColor
                        }
                        
                        Item { Layout.fillWidth: true }
                        
                        ComboBox {
                            model: ["Absolute", "Percentage", "Drawdown"]
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
                    
                    // Equity chart
                    ChartView {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        
                        backgroundColor: backgroundColor
                        titleColor: textColor
                        plotAreaColor: backgroundColor
                        
                        margins.top: 0
                        margins.bottom: 0
                        margins.left: 0
                        margins.right: 0
                        
                        legend.visible: false
                        antialiasing: true
                        
                        ValueAxis {
                            id: equityYAxis
                            min: 95000
                            max: 125000
                            tickCount: 5
                            labelFormat: "$%.0f"
                            labelsColor: secondaryTextColor
                            gridLineColor: "#30363D"
                            color: borderColor
                        }
                        
                        DateTimeAxis {
                            id: equityXAxis
                            format: "MMM dd"
                            labelsColor: secondaryTextColor
                            gridLineColor: "#30363D"
                            color: borderColor
                            tickCount: 5
                        }
                        
                        LineSeries {
                            id: equitySeries
                            name: "Equity"
                            axisX: equityXAxis
                            axisY: equityYAxis
                            color: accentGreen
                            width: 2
                            
                            Component.onCompleted: {
                                // Sample equity curve data
                                let startDate = new Date().getTime() - 30 * 24 * 3600000  // 30 days ago
                                let equity = 100000
                                
                                for (let i = 0; i < 30; i++) {
                                    let date = startDate + i * 24 * 3600000
                                    equity += (Math.random() - 0.4) * 1000  // Slight upward bias
                                    append(date, equity)
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Collapsed state
    Rectangle {
        anchors.fill: parent
        color: backgroundColor
        visible: collapsed
        opacity: collapsed ? 1 : 0
        
        Behavior on opacity {
            NumberAnimation {
                duration: animationDuration
                easing.type: Easing.OutCubic
            }
        }
        
        RowLayout {
            anchors.fill: parent
            anchors.margins: 8
            spacing: 16
            
            Text {
                text: "Status"
                font.pixelSize: 12
                font.bold: true
                color: textColor
            }
            
            Rectangle {
                width: 8
                height: 8
                radius: 4
                color: backtestManager?.isRunning ? successColor : secondaryTextColor
            }
            
            Text {
                Layout.fillWidth: true
                text: backtestManager?.statusMessage || "Ready"
                font.pixelSize: 11
                color: secondaryTextColor
                elide: Text.ElideRight
            }
            
            Text {
                text: "P&L: " + (backtestManager?.totalPnl ? "$" + backtestManager.totalPnl.toFixed(2) : "$0.00")
                font.pixelSize: 11
                font.bold: true
                color: (backtestManager?.totalPnl || 0) >= 0 ? successColor : dangerColor
            }
        }
    }
}