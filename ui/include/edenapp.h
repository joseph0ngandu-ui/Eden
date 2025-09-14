#ifndef EDENAPP_H
#define EDENAPP_H

#include <QObject>
#include <QQmlEngine>
#include <QApplication>
#include <QTimer>
#include <QLoggingCategory>

QT_BEGIN_NAMESPACE
class BacktestManager;
class GpuManager;
class WorkerManager;
class ThemeManager;
QT_END_NAMESPACE

Q_DECLARE_LOGGING_CATEGORY(edenApp)

class EdenApp : public QObject
{
    Q_OBJECT
    QML_ELEMENT
    
    Q_PROPERTY(BacktestManager* backtestManager READ backtestManager CONSTANT)
    Q_PROPERTY(GpuManager* gpuManager READ gpuManager CONSTANT)
    Q_PROPERTY(WorkerManager* workerManager READ workerManager CONSTANT)
    Q_PROPERTY(ThemeManager* themeManager READ themeManager CONSTANT)
    Q_PROPERTY(QString version READ version CONSTANT)
    Q_PROPERTY(bool isInitialized READ isInitialized NOTIFY initializedChanged)
    Q_PROPERTY(QString statusMessage READ statusMessage NOTIFY statusMessageChanged)

public:
    explicit EdenApp(QObject *parent = nullptr);
    ~EdenApp();

    // Property getters
    BacktestManager* backtestManager() const { return m_backtestManager; }
    GpuManager* gpuManager() const { return m_gpuManager; }
    WorkerManager* workerManager() const { return m_workerManager; }
    ThemeManager* themeManager() const { return m_themeManager; }
    QString version() const { return "1.0.0"; }
    bool isInitialized() const { return m_initialized; }
    QString statusMessage() const { return m_statusMessage; }

    // QML invokable methods
    Q_INVOKABLE void initialize();
    Q_INVOKABLE void shutdown();
    Q_INVOKABLE QString getApplicationDataPath() const;
    Q_INVOKABLE void showMessage(const QString &message);
    Q_INVOKABLE void exportSettings(const QString &filePath);
    Q_INVOKABLE void importSettings(const QString &filePath);

public slots:
    void setStatusMessage(const QString &message);

signals:
    void initializedChanged(bool initialized);
    void statusMessageChanged(const QString &message);
    void messageRequested(const QString &message, int timeout = 3000);
    void shutdownRequested();

private slots:
    void onInitializationStep();
    void onWorkerManagerReady();
    void onGpuManagerReady();

private:
    void initializeComponents();
    void setupConnections();
    void loadSettings();
    void saveSettings();

private:
    BacktestManager* m_backtestManager;
    GpuManager* m_gpuManager; 
    WorkerManager* m_workerManager;
    ThemeManager* m_themeManager;
    
    bool m_initialized;
    QString m_statusMessage;
    QTimer* m_initTimer;
    int m_initStep;
};

#endif // EDENAPP_H