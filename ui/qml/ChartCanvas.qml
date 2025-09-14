import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtCharts 2.15

Rectangle {
    id: chartCanvas
    
    property var backtestManager: null
    property bool showTrades: true
    property bool showLiquidityZones: true
    property bool showFVGs: true
    property bool showOrderBlocks: true
    
    // Theme colors
    readonly property color backgroundColor: "#161B22"
    readonly property color cardColor: "#21262D"
    readonly property color borderColor: "#30363D"
    readonly property color textColor: "#F0F6FC"
    readonly property color secondaryTextColor: "#8B949E"
    readonly property color accentGreen: "#238636"
    readonly property color bullishColor: "#3FB950"
    readonly property color bearishColor: "#F85149"
    readonly property color neutralColor: "#8B949E"
    
    color: backgroundColor
    border.width: 1
    border.color: borderColor
    radius: 8
    
    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 1
        spacing: 0
        
        // Chart Header with Symbol Info and Controls
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 60
            color: cardColor
            border.width: 1
            border.color: borderColor
            
            RowLayout {
                anchors.fill: parent
                anchors.margins: 16
                spacing: 16
                
                // Symbol and timeframe info
                ColumnLayout {
                    spacing: 2
                    
                    Text {
                        text: "XAUUSD"
                        font.pixelSize: 20
                        font.bold: true
                        color: textColor
                    }
                    
                    Text {
                        text: "1 Hour • Last: $2,645.23 (+0.34%)"
                        font.pixelSize: 12
                        color: bullishColor
                    }
                }
                
                Item { Layout.fillWidth: true }  // Spacer
                
                // Chart controls
                RowLayout {
                    spacing: 8
                    
                    // Timeframe selector
                    ComboBox {
                        id: timeframeCombo
                        model: ["1M", "5M", "15M", "1H", "4H", "1D", "1W"]
                        currentIndex: 3  // 1H
                        
                        delegate: ItemDelegate {
                            width: timeframeCombo.width
                            contentItem: Text {
                                text: modelData
                                color: textColor
                                font.pixelSize: 12
                                verticalAlignment: Text.AlignVCenter
                            }
                            
                            background: Rectangle {
                                color: parent.hovered ? "#30363D" : "transparent"
                            }
                        }
                        
                        background: Rectangle {
                            color: "#30363D"
                            border.color: "#484F58"
                            border.width: 1
                            radius: 6
                        }
                        
                        contentItem: Text {
                            text: timeframeCombo.displayText
                            font.pixelSize: 12
                            color: textColor
                            verticalAlignment: Text.AlignVCenter
                            leftPadding: 12
                        }
                    }
                    
                    // Chart type toggle
                    Rectangle {
                        width: 32
                        height: 32
                        radius: 6
                        color: chartTypeMouseArea.containsMouse ? "#30363D" : "transparent"
                        border.width: 1
                        border.color: "#484F58"
                        
                        Text {
                            anchors.centerIn: parent
                            text: "📊"
                            font.pixelSize: 14
                        }
                        
                        MouseArea {
                            id: chartTypeMouseArea
                            anchors.fill: parent
                            hoverEnabled: true
                            onClicked: console.log("Chart type clicked")
                        }
                    }
                    
                    // Full screen toggle
                    Rectangle {
                        width: 32
                        height: 32
                        radius: 6
                        color: fullscreenMouseArea.containsMouse ? "#30363D" : "transparent"
                        border.width: 1
                        border.color: "#484F58"
                        
                        Text {
                            anchors.centerIn: parent
                            text: "⛶"
                            font.pixelSize: 14
                            color: secondaryTextColor
                        }
                        
                        MouseArea {
                            id: fullscreenMouseArea
                            anchors.fill: parent
                            hoverEnabled: true
                            onClicked: console.log("Fullscreen clicked")
                        }
                    }
                }
            }
        }
        
        // Main Chart Area
        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            color: backgroundColor
            
            ChartView {
                id: chartView
                anchors.fill: parent
                anchors.margins: 8
                
                backgroundColor: backgroundColor
                titleColor: textColor
                plotAreaColor: backgroundColor
                
                // Remove default margins and background
                margins.top: 0
                margins.bottom: 0
                margins.left: 0
                margins.right: 0
                
                legend.visible: false
                antialiasing: true
                
                // Price axis (right side)
                ValueAxis {
                    id: priceAxis
                    min: 2600
                    max: 2700
                    tickCount: 6
                    labelFormat: "%.2f"
                    labelsColor: secondaryTextColor
                    gridLineColor: "#30363D"
                    color: borderColor
                }
                
                // Time axis (bottom)
                DateTimeAxis {
                    id: timeAxis
                    format: "MMM dd hh:mm"
                    labelsColor: secondaryTextColor
                    gridLineColor: "#30363D"
                    color: borderColor
                    tickCount: 6
                }
                
                // Candlestick series
                CandlestickSeries {
                    id: candlestickSeries
                    name: "XAUUSD"
                    axisX: timeAxis
                    axisY: priceAxis
                    
                    increasingColor: bullishColor
                    decreasingColor: bearishColor
                    bodyWidth: 0.8
                    capsWidth: 0.5
                    
                    // Sample data - in production this would come from backtestManager
                    Component.onCompleted: {
                        // Add sample candlestick data
                        for (let i = 0; i < 50; i++) {
                            let timestamp = new Date().getTime() - (50 - i) * 3600000  // 1 hour intervals
                            let open = 2650 + Math.random() * 20 - 10
                            let high = open + Math.random() * 15
                            let low = open - Math.random() * 15
                            let close = open + (Math.random() - 0.5) * 10
                            
                            candlestickSeries.append(timestamp, open, high, low, close)
                        }
                    }
                }
                
                // Volume series (if needed)
                // LineSeries for MA/EMA overlays could be added here
            }
            
            // Trade markers overlay
            Repeater {
                id: tradeMarkers
                model: []  // Will be populated by backtestManager
                
                delegate: Rectangle {
                    property var trade: modelData
                    x: chartCanvas.mapTimeToX(trade.timestamp)
                    y: chartCanvas.mapPriceToY(trade.price)
                    width: 12
                    height: 12
                    radius: 6
                    color: trade.side === "buy" ? bullishColor : bearishColor
                    border.width: 2
                    border.color: backgroundColor
                    
                    // Entry/Exit marker
                    Rectangle {
                        anchors.centerIn: parent
                        width: 4
                        height: 4
                        radius: 2
                        color: "white"
                    }
                    
                    // Tooltip on hover
                    MouseArea {
                        anchors.fill: parent
                        hoverEnabled: true
                        
                        ToolTip {
                            visible: parent.containsMouse
                            text: `${trade.side.toUpperCase()} @ ${trade.price.toFixed(5)}\n${new Date(trade.timestamp).toLocaleDateString()}`
                            background: Rectangle {
                                color: cardColor
                                border.color: borderColor
                                border.width: 1
                                radius: 6
                            }
                            contentItem: Text {
                                text: parent.text
                                color: textColor
                                font.pixelSize: 11
                            }
                        }
                    }
                }
            }
            
            // Liquidity zones overlay (ICT concepts)
            Canvas {
                id: liquidityCanvas
                anchors.fill: parent
                visible: showLiquidityZones
                
                onPaint: {
                    let ctx = getContext("2d")
                    ctx.clearRect(0, 0, width, height)
                    
                    // Draw sample liquidity zones
                    ctx.globalAlpha = 0.2
                    ctx.fillStyle = bullishColor
                    ctx.fillRect(50, height * 0.3, width - 100, 30)
                    
                    ctx.fillStyle = bearishColor
                    ctx.fillRect(50, height * 0.7, width - 100, 30)
                    
                    ctx.globalAlpha = 1.0
                }
            }
            
            // Fair Value Gaps overlay
            Canvas {
                id: fvgCanvas
                anchors.fill: parent
                visible: showFVGs
                
                onPaint: {
                    let ctx = getContext("2d")
                    ctx.clearRect(0, 0, width, height)
                    
                    // Draw sample FVGs
                    ctx.strokeStyle = accentGreen
                    ctx.lineWidth = 2
                    ctx.setLineDash([5, 5])
                    
                    ctx.beginPath()
                    ctx.moveTo(width * 0.3, height * 0.4)
                    ctx.lineTo(width * 0.8, height * 0.4)
                    ctx.stroke()
                    
                    ctx.beginPath()
                    ctx.moveTo(width * 0.3, height * 0.6)
                    ctx.lineTo(width * 0.8, height * 0.6)
                    ctx.stroke()
                }
            }
        }
        
        // Chart indicators legend/overlay controls
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 40
            color: cardColor
            border.width: 1
            border.color: borderColor
            
            RowLayout {
                anchors.fill: parent
                anchors.margins: 12
                spacing: 16
                
                // Overlay toggles
                Row {
                    spacing: 12
                    
                    CheckBox {
                        id: tradesCheck
                        text: "Trades"
                        checked: showTrades
                        onCheckedChanged: showTrades = checked
                        
                        contentItem: Text {
                            text: parent.text
                            font.pixelSize: 12
                            color: textColor
                            leftPadding: parent.indicator.width + parent.spacing
                            verticalAlignment: Text.AlignVCenter
                        }
                    }
                    
                    CheckBox {
                        id: liquidityCheck
                        text: "Liquidity"
                        checked: showLiquidityZones
                        onCheckedChanged: showLiquidityZones = checked
                        
                        contentItem: Text {
                            text: parent.text
                            font.pixelSize: 12
                            color: textColor
                            leftPadding: parent.indicator.width + parent.spacing
                            verticalAlignment: Text.AlignVCenter
                        }
                    }
                    
                    CheckBox {
                        id: fvgCheck
                        text: "FVGs"
                        checked: showFVGs
                        onCheckedChanged: showFVGs = checked
                        
                        contentItem: Text {
                            text: parent.text
                            font.pixelSize: 12
                            color: textColor
                            leftPadding: parent.indicator.width + parent.spacing
                            verticalAlignment: Text.AlignVCenter
                        }
                    }
                }
                
                Item { Layout.fillWidth: true }
                
                // Chart stats
                Text {
                    text: "Last Updated: " + new Date().toLocaleTimeString()
                    font.pixelSize: 11
                    color: secondaryTextColor
                }
            }
        }
    }
    
    // Utility functions for coordinate mapping
    function mapTimeToX(timestamp) {
        // Convert timestamp to chart X coordinate
        return chartView.plotArea.x + (chartView.plotArea.width * 0.5)  // Placeholder
    }
    
    function mapPriceToY(price) {
        // Convert price to chart Y coordinate
        return chartView.plotArea.y + (chartView.plotArea.height * 0.5)  // Placeholder
    }
    
    // Update chart data when backtest manager changes
    Connections {
        target: backtestManager
        function onNewCandle(candleData) {
            // Add new candle to chart
            candlestickSeries.append(candleData.timestamp, candleData.open, 
                                   candleData.high, candleData.low, candleData.close)
        }
        
        function onTradeExecuted(trade) {
            // Add trade marker
            tradeMarkers.model.push(trade)
        }
    }
}