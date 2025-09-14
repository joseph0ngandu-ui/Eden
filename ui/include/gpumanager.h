#ifndef GPUMANAGER_H
#define GPUMANAGER_H

#include <QObject>
#include <QQmlEngine>
#include <QString>
#include <QStringList>
#include <QTimer>
#include <QLoggingCategory>
#include <QProcess>

#ifdef ONNX_AVAILABLE
#include <onnxruntime_cxx_api.h>
#endif

Q_DECLARE_LOGGING_CATEGORY(gpuManager)

class GpuManager : public QObject
{
    Q_OBJECT
    QML_ELEMENT
    
    Q_PROPERTY(bool isEnabled READ isEnabled WRITE setEnabled NOTIFY isEnabledChanged)
    Q_PROPERTY(bool isAvailable READ isAvailable NOTIFY isAvailableChanged)
    Q_PROPERTY(QString currentProvider READ currentProvider NOTIFY currentProviderChanged)
    Q_PROPERTY(QString preferredProvider READ preferredProvider WRITE setPreferredProvider NOTIFY preferredProviderChanged)
    Q_PROPERTY(QStringList availableProviders READ availableProviders NOTIFY availableProvidersChanged)
    Q_PROPERTY(QString deviceName READ deviceName NOTIFY deviceNameChanged)
    Q_PROPERTY(QString statusText READ statusText NOTIFY statusTextChanged)
    Q_PROPERTY(int memoryUsed READ memoryUsed NOTIFY memoryUsedChanged)
    Q_PROPERTY(int memoryTotal READ memoryTotal NOTIFY memoryTotalChanged)
    Q_PROPERTY(double memoryUsagePercent READ memoryUsagePercent NOTIFY memoryUsagePercentChanged)
    Q_PROPERTY(bool isInitialized READ isInitialized NOTIFY isInitializedChanged)

public:
    explicit GpuManager(QObject *parent = nullptr);
    ~GpuManager();

    // Property getters
    bool isEnabled() const { return m_isEnabled; }
    bool isAvailable() const { return m_isAvailable; }
    QString currentProvider() const { return m_currentProvider; }
    QString preferredProvider() const { return m_preferredProvider; }
    QStringList availableProviders() const { return m_availableProviders; }
    QString deviceName() const { return m_deviceName; }
    QString statusText() const { return m_statusText; }
    int memoryUsed() const { return m_memoryUsed; }
    int memoryTotal() const { return m_memoryTotal; }
    double memoryUsagePercent() const { return m_memoryUsagePercent; }
    bool isInitialized() const { return m_isInitialized; }

    // Property setters
    void setEnabled(bool enabled);
    void setPreferredProvider(const QString &provider);

    // QML invokable methods
    Q_INVOKABLE void initialize();
    Q_INVOKABLE void shutdown();
    Q_INVOKABLE void refreshStatus();
    Q_INVOKABLE void switchProvider(const QString &provider);
    Q_INVOKABLE QStringList detectGpuDevices();
    Q_INVOKABLE QString getGpuInfo();
    Q_INVOKABLE bool testGpuPerformance();
    Q_INVOKABLE void clearGpuMemory();

public slots:
    void onWorkerStatusChanged();

signals:
    void isEnabledChanged(bool enabled);
    void isAvailableChanged(bool available);
    void currentProviderChanged(const QString &provider);
    void preferredProviderChanged(const QString &provider);
    void availableProvidersChanged(const QStringList &providers);
    void deviceNameChanged(const QString &name);
    void statusTextChanged(const QString &status);
    void memoryUsedChanged(int used);
    void memoryTotalChanged(int total);
    void memoryUsagePercentChanged(double percent);
    void isInitializedChanged(bool initialized);
    
    void ready();
    void statusMessage(const QString &message);
    void errorOccurred(const QString &error);

private slots:
    void updateMemoryUsage();
    void checkProviderAvailability();

private:
    void detectAvailableProviders();
    void initializeOnnxRuntime();
    void shutdownOnnxRuntime();
    bool initializeProvider(const QString &provider);
    void updateStatus();
    void queryGpuMemory();
    QString getProviderDisplayName(const QString &provider) const;
    bool isProviderAvailable(const QString &provider) const;

private:
    // Configuration
    bool m_isEnabled;
    bool m_isAvailable;
    bool m_isInitialized;
    QString m_currentProvider;
    QString m_preferredProvider;
    QStringList m_availableProviders;
    
    // Device info
    QString m_deviceName;
    QString m_statusText;
    int m_memoryUsed;
    int m_memoryTotal;
    double m_memoryUsagePercent;
    
    // Timers
    QTimer* m_memoryTimer;
    QTimer* m_statusTimer;
    
#ifdef ONNX_AVAILABLE
    // ONNX Runtime session
    std::unique_ptr<Ort::Env> m_ortEnv;
    std::unique_ptr<Ort::SessionOptions> m_sessionOptions;
    std::unique_ptr<Ort::Session> m_session;
#endif
};

#endif // GPUMANAGER_H