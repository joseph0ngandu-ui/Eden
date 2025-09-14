#ifndef BACKTESTMANAGER_H
#define BACKTESTMANAGER_H

#include <QObject>
#include <QQmlEngine>
#include <QString>
#include <QVariantMap>
#include <QTimer>
#include <QLoggingCategory>
#include <QSqlDatabase>
#include <QJsonObject>
#include <QJsonDocument>
#include <QDateTime>

QT_BEGIN_NAMESPACE
class WorkerManager;
class GpuManager;
QT_END_NAMESPACE

Q_DECLARE_LOGGING_CATEGORY(backtestManager)

class BacktestManager : public QObject
{
    Q_OBJECT
    QML_ELEMENT
    
    Q_PROPERTY(bool isRunning READ isRunning NOTIFY isRunningChanged)
    Q_PROPERTY(double progress READ progress NOTIFY progressChanged)
    Q_PROPERTY(QString statusMessage READ statusMessage NOTIFY statusMessageChanged)
    Q_PROPERTY(double totalPnl READ totalPnl NOTIFY totalPnlChanged)
    Q_PROPERTY(int totalTrades READ totalTrades NOTIFY totalTradesChanged)
    Q_PROPERTY(double winRate READ winRate NOTIFY winRateChanged)
    Q_PROPERTY(QString currentBacktestId READ currentBacktestId NOTIFY currentBacktestIdChanged)
    Q_PROPERTY(QVariantMap currentResults READ currentResults NOTIFY currentResultsChanged)

public:
    explicit BacktestManager(QObject *parent = nullptr);
    ~BacktestManager();

    // Property getters
    bool isRunning() const { return m_isRunning; }
    double progress() const { return m_progress; }
    QString statusMessage() const { return m_statusMessage; }
    double totalPnl() const { return m_totalPnl; }
    int totalTrades() const { return m_totalTrades; }
    double winRate() const { return m_winRate; }
    QString currentBacktestId() const { return m_currentBacktestId; }
    QVariantMap currentResults() const { return m_currentResults; }

    // Dependencies
    void setWorkerManager(WorkerManager* workerManager);
    void setGpuManager(GpuManager* gpuManager);

    // QML invokable methods
    Q_INVOKABLE void runBacktest(const QVariantMap &parameters);
    Q_INVOKABLE void stopBacktest();
    Q_INVOKABLE void stopAllBacktests();
    Q_INVOKABLE QVariantList getBacktestHistory(int limit = 50);
    Q_INVOKABLE QVariantMap getBacktestResults(const QString &backtestId);
    Q_INVOKABLE bool deleteBacktest(const QString &backtestId);
    Q_INVOKABLE QString exportBacktestResults(const QString &backtestId, const QString &format = "json");
    Q_INVOKABLE QVariantList compareBacktests(const QStringList &backtestIds);
    Q_INVOKABLE void optimizeStrategy(const QVariantMap &parameters);

public slots:
    void onBacktestCompleted(const QString &requestId, const QVariantMap &results);
    void onBacktestProgress(const QString &requestId, double progress, const QString &message);
    void onBacktestError(const QString &requestId, const QString &error);

signals:
    void isRunningChanged(bool isRunning);
    void progressChanged(double progress);
    void statusMessageChanged(const QString &message);
    void totalPnlChanged(double totalPnl);
    void totalTradesChanged(int totalTrades);
    void winRateChanged(double winRate);
    void currentBacktestIdChanged(const QString &backtestId);
    void currentResultsChanged(const QVariantMap &results);
    
    void backtestCompleted(const QString &backtestId, const QVariantMap &results);
    void backtestStarted(const QString &backtestId);
    void backtestStopped(const QString &backtestId);
    void backtestError(const QString &backtestId, const QString &error);

private slots:
    void updateMetrics();
    void checkBacktestStatus();

private:
    void initializeDatabase();
    QString generateBacktestId();
    void saveBacktestResults(const QString &backtestId, const QVariantMap &results);
    void loadBacktestResults(const QString &backtestId);
    void updateCurrentMetrics(const QVariantMap &results);
    QString createBacktestDirectory(const QString &backtestId);
    void saveResultsToFiles(const QString &backtestId, const QVariantMap &results);
    QJsonObject getReproducibilityMetadata();

private:
    WorkerManager* m_workerManager;
    GpuManager* m_gpuManager;
    
    // State
    bool m_isRunning;
    double m_progress;
    QString m_statusMessage;
    QString m_currentBacktestId;
    QVariantMap m_currentResults;
    
    // Metrics
    double m_totalPnl;
    int m_totalTrades;
    double m_winRate;
    
    // Database
    QSqlDatabase m_database;
    
    // Timers
    QTimer* m_metricsTimer;
    QTimer* m_statusTimer;
};

#endif // BACKTESTMANAGER_H