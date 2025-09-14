#ifndef THEMEMANAGER_H
#define THEMEMANAGER_H

#include <QObject>
#include <QQmlEngine>
#include <QString>
#include <QColor>
#include <QVariantMap>
#include <QLoggingCategory>

Q_DECLARE_LOGGING_CATEGORY(themeManager)

class ThemeManager : public QObject
{
    Q_OBJECT
    QML_ELEMENT
    
    Q_PROPERTY(QString currentTheme READ currentTheme WRITE setCurrentTheme NOTIFY currentThemeChanged)
    Q_PROPERTY(QVariantMap colors READ colors NOTIFY colorsChanged)
    Q_PROPERTY(bool isDarkTheme READ isDarkTheme NOTIFY isDarkThemeChanged)
    Q_PROPERTY(QStringList availableThemes READ availableThemes CONSTANT)

public:
    explicit ThemeManager(QObject *parent = nullptr);
    ~ThemeManager();

    // Property getters
    QString currentTheme() const { return m_currentTheme; }
    QVariantMap colors() const { return m_colors; }
    bool isDarkTheme() const { return m_isDarkTheme; }
    QStringList availableThemes() const { return m_availableThemes; }

    // Property setters
    void setCurrentTheme(const QString &theme);

    // QML invokable methods
    Q_INVOKABLE QColor getColor(const QString &colorName) const;
    Q_INVOKABLE void setColor(const QString &colorName, const QColor &color);
    Q_INVOKABLE void resetTheme();
    Q_INVOKABLE void exportTheme(const QString &filePath);
    Q_INVOKABLE void importTheme(const QString &filePath);

signals:
    void currentThemeChanged(const QString &theme);
    void colorsChanged(const QVariantMap &colors);
    void isDarkThemeChanged(bool isDark);

private:
    void loadTheme(const QString &themeName);
    void initializeDefaultThemes();
    void updateColors();
    QVariantMap createDarkTheme();
    QVariantMap createLightTheme();

private:
    QString m_currentTheme;
    QVariantMap m_colors;
    bool m_isDarkTheme;
    QStringList m_availableThemes;
    QMap<QString, QVariantMap> m_themeData;
};

#endif // THEMEMANAGER_H