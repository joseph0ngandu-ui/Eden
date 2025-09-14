#include "edenapp.h"
#include "backtestmanager.h"
#include "gpumanager.h"
#include "workermanager.h"
#include "thememanager.h"

#include <QStandardPaths>
#include <QDir>
#include <QSettings>
#include <QJsonDocument>
#include <QJsonObject>
#include <QDebug>
#include <QApplication>

Q_LOGGING_CATEGORY(edenApp, "eden.app")

EdenApp::EdenApp(QObject *parent)
    : QObject(parent)
    , m_backtestManager(nullptr)
    , m_gpuManager(nullptr)
    , m_workerManager(nullptr)
    , m_themeManager(nullptr)
    , m_initialized(false)
    , m_statusMessage("Initializing...")
    , m_initTimer(new QTimer(this))
    , m_initStep(0)
{
    qCDebug(edenApp) << "Creating EdenApp instance";
    
    // Initialize timer for staged initialization
    m_initTimer->setSingleShot(true);
    connect(m_initTimer, &QTimer::timeout, this, &EdenApp::onInitializationStep);
    
    // Start initialization process
    QTimer::singleShot(100, this, &EdenApp::initialize);
}

EdenApp::~EdenApp()
{
    qCDebug(edenApp) << "Destroying EdenApp instance";
    saveSettings();
}

void EdenApp::initialize()
{
    qCDebug(edenApp) << "Starting Eden application initialization";
    
    if (m_initialized) {
        qCWarning(edenApp) << "Already initialized";
        return;
    }
    
    setStatusMessage("Initializing core components...");
    
    // Create application directories
    QString appDataPath = getApplicationDataPath();
    QDir().mkpath(appDataPath);
    QDir().mkpath(appDataPath + "/backtests");
    QDir().mkpath(appDataPath + "/logs");
    QDir().mkpath(appDataPath + "/models");
    QDir().mkpath(appDataPath + "/temp");
    
    // Start staged initialization
    m_initStep = 0;
    m_initTimer->start(200);
}

void EdenApp::onInitializationStep()
{
    switch (m_initStep) {
    case 0:
        setStatusMessage("Initializing theme manager...");
        m_themeManager = new ThemeManager(this);
        break;
        
    case 1:
        setStatusMessage("Initializing GPU acceleration...");
        m_gpuManager = new GpuManager(this);
        connect(m_gpuManager, &GpuManager::ready, this, &EdenApp::onGpuManagerReady);
        m_gpuManager->initialize();
        break;
        
    case 2:
        setStatusMessage("Initializing Python workers...");
        m_workerManager = new WorkerManager(this);
        connect(m_workerManager, &WorkerManager::ready, this, &EdenApp::onWorkerManagerReady);
        m_workerManager->initialize();
        break;
        
    case 3:
        setStatusMessage("Initializing backtest manager...");
        m_backtestManager = new BacktestManager(this);
        m_backtestManager->setWorkerManager(m_workerManager);
        m_backtestManager->setGpuManager(m_gpuManager);
        break;
        
    case 4:
        setupConnections();
        loadSettings();
        setStatusMessage("Ready");
        m_initialized = true;
        emit initializedChanged(true);
        qCDebug(edenApp) << "Eden application initialized successfully";
        return;
        
    default:
        qCWarning(edenApp) << "Unknown initialization step:" << m_initStep;
        return;
    }
    
    m_initStep++;
    m_initTimer->start(300);
}

void EdenApp::onWorkerManagerReady()
{
    qCDebug(edenApp) << "Worker manager ready";
    // Continue with next step if this was the current bottleneck
}

void EdenApp::onGpuManagerReady()
{
    qCDebug(edenApp) << "GPU manager ready";
    // Continue with next step if this was the current bottleneck
}

void EdenApp::setupConnections()
{
    // Connect worker manager to backtest manager
    if (m_workerManager && m_backtestManager) {
        connect(m_workerManager, &WorkerManager::backtestCompleted,
                m_backtestManager, &BacktestManager::onBacktestCompleted);
        connect(m_workerManager, &WorkerManager::backtestProgress,
                m_backtestManager, &BacktestManager::onBacktestProgress);
        connect(m_workerManager, &WorkerManager::backtestError,
                m_backtestManager, &BacktestManager::onBacktestError);
    }
    
    // Connect status messages
    if (m_workerManager) {
        connect(m_workerManager, &WorkerManager::statusMessage,
                this, &EdenApp::setStatusMessage);
    }
    if (m_gpuManager) {
        connect(m_gpuManager, &GpuManager::statusMessage,
                this, &EdenApp::setStatusMessage);
    }
    if (m_backtestManager) {
        connect(m_backtestManager, &BacktestManager::statusMessage,
                this, &EdenApp::setStatusMessage);
    }
}

void EdenApp::shutdown()
{
    qCDebug(edenApp) << "Shutting down Eden application";
    
    setStatusMessage("Shutting down...");
    
    if (m_backtestManager) {
        m_backtestManager->stopAllBacktests();
    }
    
    if (m_workerManager) {
        m_workerManager->shutdown();
    }
    
    if (m_gpuManager) {
        m_gpuManager->shutdown();
    }
    
    saveSettings();
    
    emit shutdownRequested();
    QApplication::quit();
}

QString EdenApp::getApplicationDataPath() const
{
    return QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
}

void EdenApp::showMessage(const QString &message)
{
    qCInfo(edenApp) << "Message:" << message;
    emit messageRequested(message);
}

void EdenApp::setStatusMessage(const QString &message)
{
    if (m_statusMessage != message) {
        m_statusMessage = message;
        emit statusMessageChanged(message);
        qCDebug(edenApp) << "Status:" << message;
    }
}

void EdenApp::loadSettings()
{
    QSettings settings;
    
    // Load theme settings
    if (m_themeManager) {
        QString theme = settings.value("theme", "dark").toString();
        m_themeManager->setCurrentTheme(theme);
    }
    
    // Load GPU settings
    if (m_gpuManager) {
        bool gpuEnabled = settings.value("gpu/enabled", true).toBool();
        QString preferredProvider = settings.value("gpu/preferredProvider", "auto").toString();
        m_gpuManager->setEnabled(gpuEnabled);
        m_gpuManager->setPreferredProvider(preferredProvider);
    }
    
    qCDebug(edenApp) << "Settings loaded";
}

void EdenApp::saveSettings()
{
    QSettings settings;
    
    // Save theme settings
    if (m_themeManager) {
        settings.setValue("theme", m_themeManager->currentTheme());
    }
    
    // Save GPU settings
    if (m_gpuManager) {
        settings.setValue("gpu/enabled", m_gpuManager->isEnabled());
        settings.setValue("gpu/preferredProvider", m_gpuManager->preferredProvider());
    }
    
    // Save window geometry and state
    settings.setValue("version", version());
    
    qCDebug(edenApp) << "Settings saved";
}

void EdenApp::exportSettings(const QString &filePath)
{
    QSettings settings;
    QJsonObject json;
    
    // Export all settings to JSON
    for (const QString &key : settings.allKeys()) {
        json[key] = QJsonValue::fromVariant(settings.value(key));
    }
    
    QJsonDocument doc(json);
    QFile file(filePath);
    if (file.open(QIODevice::WriteOnly)) {
        file.write(doc.toJson());
        showMessage("Settings exported successfully");
    } else {
        showMessage("Failed to export settings");
    }
}

void EdenApp::importSettings(const QString &filePath)
{
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly)) {
        showMessage("Failed to read settings file");
        return;
    }
    
    QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
    if (!doc.isObject()) {
        showMessage("Invalid settings file format");
        return;
    }
    
    QSettings settings;
    QJsonObject json = doc.object();
    
    for (auto it = json.begin(); it != json.end(); ++it) {
        settings.setValue(it.key(), it.value().toVariant());
    }
    
    loadSettings();
    showMessage("Settings imported successfully");
}