#ifndef CHARTCANVAS_H
#define CHARTCANVAS_H

#include <QObject>
#include <QQmlEngine>
#include <QString>
#include <QVariantMap>
#include <QVariantList>
#include <QLoggingCategory>

Q_DECLARE_LOGGING_CATEGORY(chartCanvas)

class ChartCanvas : public QObject
{
    Q_OBJECT
    QML_ELEMENT
    
    Q_PROPERTY(QString symbol READ symbol WRITE setSymbol NOTIFY symbolChanged)
    Q_PROPERTY(QString timeframe READ timeframe WRITE setTimeframe NOTIFY timeframeChanged)
    Q_PROPERTY(QVariantList candleData READ candleData NOTIFY candleDataChanged)
    Q_PROPERTY(QVariantList tradeMarkers READ tradeMarkers NOTIFY tradeMarkersChanged)
    Q_PROPERTY(bool showTrades READ showTrades WRITE setShowTrades NOTIFY showTradesChanged)
    Q_PROPERTY(bool showLiquidity READ showLiquidity WRITE setShowLiquidity NOTIFY showLiquidityChanged)
    Q_PROPERTY(bool showFVGs READ showFVGs WRITE setShowFVGs NOTIFY showFVGsChanged)

public:
    explicit ChartCanvas(QObject *parent = nullptr);
    ~ChartCanvas();

    // Property getters
    QString symbol() const { return m_symbol; }
    QString timeframe() const { return m_timeframe; }
    QVariantList candleData() const { return m_candleData; }
    QVariantList tradeMarkers() const { return m_tradeMarkers; }
    bool showTrades() const { return m_showTrades; }
    bool showLiquidity() const { return m_showLiquidity; }
    bool showFVGs() const { return m_showFVGs; }

    // Property setters
    void setSymbol(const QString &symbol);
    void setTimeframe(const QString &timeframe);
    void setShowTrades(bool show);
    void setShowLiquidity(bool show);
    void setShowFVGs(bool show);

    // QML invokable methods
    Q_INVOKABLE void loadChartData(const QString &symbol, const QString &timeframe);
    Q_INVOKABLE void addTradeMarker(const QVariantMap &trade);
    Q_INVOKABLE void clearTradeMarkers();
    Q_INVOKABLE void zoomToFit();
    Q_INVOKABLE void exportChart(const QString &filePath, const QString &format = "png");

public slots:
    void onNewCandle(const QVariantMap &candleData);
    void onTradeExecuted(const QVariantMap &trade);
    void onBacktestDataChanged(const QVariantList &data);

signals:
    void symbolChanged(const QString &symbol);
    void timeframeChanged(const QString &timeframe);
    void candleDataChanged(const QVariantList &data);
    void tradeMarkersChanged(const QVariantList &markers);
    void showTradesChanged(bool show);
    void showLiquidityChanged(bool show);
    void showFVGsChanged(bool show);

private:
    void updateChart();
    void processChartData(const QVariantList &rawData);

private:
    QString m_symbol;
    QString m_timeframe;
    QVariantList m_candleData;
    QVariantList m_tradeMarkers;
    bool m_showTrades;
    bool m_showLiquidity;
    bool m_showFVGs;
};

#endif // CHARTCANVAS_H