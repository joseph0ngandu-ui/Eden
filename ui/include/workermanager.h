#ifndef WORKERMANAGER_H
#define WORKERMANAGER_H

#include <QObject>
#include <QQmlEngine>
#include <QString>
#include <QVariantMap>
#include <QTimer>
#include <QProcess>
#include <QLoggingCategory>
#include <QThread>
#include <QMutex>
#include <QQueue>
#include <QJsonObject>
#include <QUuid>

QT_BEGIN_NAMESPACE
class QNetworkAccessManager;
QT_END_NAMESPACE

// Forward declarations for ZeroMQ (conditionally compiled)
#ifdef ZMQ_AVAILABLE
namespace zmq {
    class context_t;
    class socket_t;
}
#endif

Q_DECLARE_LOGGING_CATEGORY(workerManager)

class WorkerManager : public QObject
{
    Q_OBJECT
    QML_ELEMENT
    
    Q_PROPERTY(bool isConnected READ isConnected NOTIFY isConnectedChanged)
    Q_PROPERTY(bool isWorkerRunning READ isWorkerRunning NOTIFY isWorkerRunningChanged)
    Q_PROPERTY(QString workerStatus READ workerStatus NOTIFY workerStatusChanged)
    Q_PROPERTY(int activeTasks READ activeTasks NOTIFY activeTasksChanged)
    Q_PROPERTY(QString lastError READ lastError NOTIFY lastErrorChanged)
    Q_PROPERTY(bool isInitialized READ isInitialized NOTIFY isInitializedChanged)

public:
    explicit WorkerManager(QObject *parent = nullptr);
    ~WorkerManager();

    // Property getters
    bool isConnected() const { return m_isConnected; }
    bool isWorkerRunning() const { return m_isWorkerRunning; }
    QString workerStatus() const { return m_workerStatus; }
    int activeTasks() const { return m_activeTasks; }
    QString lastError() const { return m_lastError; }
    bool isInitialized() const { return m_isInitialized; }

    // QML invokable methods
    Q_INVOKABLE void initialize();
    Q_INVOKABLE void shutdown();
    Q_INVOKABLE void startWorker();
    Q_INVOKABLE void stopWorker();
    Q_INVOKABLE void restartWorker();
    Q_INVOKABLE bool pingWorker();
    Q_INVOKABLE void sendCommand(const QString &command, const QVariantMap &params = QVariantMap());
    Q_INVOKABLE QVariantMap getWorkerStatus();
    Q_INVOKABLE void clearTaskQueue();

    // Internal API for other managers
    QString sendCommandSync(const QString &command, const QVariantMap &params, int timeoutMs = 5000);
    void sendCommandAsync(const QString &command, const QVariantMap &params, const QString &requestId = QString());

public slots:
    void onWorkerProcessFinished(int exitCode, QProcess::ExitStatus exitStatus);
    void onWorkerProcessError(QProcess::ProcessError error);
    void onWorkerProcessStarted();

signals:
    void isConnectedChanged(bool connected);
    void isWorkerRunningChanged(bool running);
    void workerStatusChanged(const QString &status);
    void activeTasksChanged(int tasks);
    void lastErrorChanged(const QString &error);
    void isInitializedChanged(bool initialized);
    
    void ready();
    void statusMessage(const QString &message);
    void errorOccurred(const QString &error);
    
    // Backtest-specific signals
    void backtestCompleted(const QString &requestId, const QVariantMap &results);
    void backtestProgress(const QString &requestId, double progress, const QString &message);
    void backtestError(const QString &requestId, const QString &error);
    
    // Worker lifecycle signals
    void workerStarted();
    void workerStopped();
    void workerConnected();
    void workerDisconnected();

private slots:
    void checkConnection();
    void processMessageQueue();
    void handleProgressUpdate();

private:
    // ZeroMQ communication
    void initializeZmq();
    void shutdownZmq();
    void connectToWorker();
    void disconnectFromWorker();
    bool sendZmqMessage(const QJsonObject &message);
    QJsonObject receiveZmqMessage(int timeoutMs = 1000);
    void processSubscriptionMessages();
    
    // Worker process management
    void startWorkerProcess();
    void stopWorkerProcess();
    QString getWorkerExecutablePath();
    QStringList getWorkerArguments();
    
    // Message handling
    QString generateRequestId();
    void addPendingRequest(const QString &requestId, const QString &command);
    void removePendingRequest(const QString &requestId);
    void handleResponse(const QJsonObject &response);
    void handleProgressMessage(const QJsonObject &message);
    void handleStatusMessage(const QJsonObject &message);
    
    // Utilities
    void updateStatus();
    void setLastError(const QString &error);

private:
    // Connection state
    bool m_isConnected;
    bool m_isWorkerRunning;
    bool m_isInitialized;
    QString m_workerStatus;
    int m_activeTasks;
    QString m_lastError;
    
    // Worker process
    QProcess* m_workerProcess;
    QString m_workerPath;
    
    // ZeroMQ sockets
#ifdef ZMQ_AVAILABLE
    std::unique_ptr<zmq::context_t> m_zmqContext;
    std::unique_ptr<zmq::socket_t> m_reqSocket;      // REQ socket for commands
    std::unique_ptr<zmq::socket_t> m_subSocket;      // SUB socket for progress updates
#endif
    
    // Communication settings
    int m_reqPort;
    int m_subPort;
    QString m_workerHost;
    
    // Message tracking
    QMutex m_requestMutex;
    QMap<QString, QPair<QString, QDateTime>> m_pendingRequests; // requestId -> (command, timestamp)
    QQueue<QJsonObject> m_messageQueue;
    
    // Timers
    QTimer* m_connectionTimer;
    QTimer* m_messageTimer;
    
    // Background thread for ZMQ subscription handling
    QThread* m_subThread;
};

// Background thread class for handling ZeroMQ subscriptions
class ZmqSubscriptionWorker : public QObject
{
    Q_OBJECT

public:
    explicit ZmqSubscriptionWorker(WorkerManager* manager, QObject* parent = nullptr);
    ~ZmqSubscriptionWorker();

public slots:
    void startListening();
    void stopListening();

signals:
    void messageReceived(const QJsonObject &message);
    void errorOccurred(const QString &error);

private:
    WorkerManager* m_manager;
    bool m_listening;
    
#ifdef ZMQ_AVAILABLE
    std::unique_ptr<zmq::context_t> m_context;
    std::unique_ptr<zmq::socket_t> m_socket;
#endif
};

#endif // WORKERMANAGER_H