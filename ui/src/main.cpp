#include <QApplication>
#include <QQmlApplicationEngine>
#include <QQuickStyle>
#include <QDir>
#include <QStandardPaths>
#include <QLoggingCategory>
#include <QIcon>

#include "edenapp.h"
#include "backtestmanager.h"
#include "gpumanager.h"
#include "workermanager.h"
#include "chartcanvas.h"
#include "thememanager.h"

Q_LOGGING_CATEGORY(edenMain, "eden.main")

int main(int argc, char *argv[])
{
    // Enable high DPI scaling
    QApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    QApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);
    
    QApplication app(argc, argv);
    
    // Set application properties
    app.setApplicationName("Eden");
    app.setApplicationVersion("1.0.0");
    app.setApplicationDisplayName("Eden Trading System");
    app.setOrganizationName("Eden Trading");
    app.setOrganizationDomain("eden.trading");
    
    // Set application icon
    app.setWindowIcon(QIcon(":/resources/icons/eden-icon.png"));
    
    // Set QuickStyle for modern appearance
    QQuickStyle::setStyle("Material");
    
    // Initialize logging
    QLoggingCategory::setFilterRules("eden.*=true");
    qCDebug(edenMain) << "Starting Eden Trading System v1.0.0";
    
    // Create application directories
    QString appDataPath = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    QDir().mkpath(appDataPath);
    QDir().mkpath(appDataPath + "/backtests");
    QDir().mkpath(appDataPath + "/logs");
    QDir().mkpath(appDataPath + "/models");
    
    // Register QML types
    qmlRegisterType<EdenApp>("Eden", 1, 0, "EdenApp");
    qmlRegisterType<BacktestManager>("Eden", 1, 0, "BacktestManager");
    qmlRegisterType<GpuManager>("Eden", 1, 0, "GpuManager");
    qmlRegisterType<WorkerManager>("Eden", 1, 0, "WorkerManager");
    qmlRegisterType<ChartCanvas>("Eden", 1, 0, "ChartCanvas");
    qmlRegisterType<ThemeManager>("Eden", 1, 0, "ThemeManager");
    
    // Create QML engine
    QQmlApplicationEngine engine;
    
    // Create and register main application object
    EdenApp edenApp;
    engine.rootContext()->setContextProperty("edenApp", &edenApp);
    
    // Load main QML file
    const QUrl url(QStringLiteral("qrc:/qml/main.qml"));
    QObject::connect(&engine, &QQmlApplicationEngine::objectCreated,
                     &app, [url](QObject *obj, const QUrl &objUrl) {
        if (!obj && url == objUrl)
            QCoreApplication::exit(-1);
    }, Qt::QueuedConnection);
    
    engine.load(url);
    
    qCDebug(edenMain) << "Eden UI loaded successfully";
    
    return app.exec();
}