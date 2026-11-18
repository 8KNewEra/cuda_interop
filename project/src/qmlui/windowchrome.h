#ifndef WINDOWCHROME_H
#define WINDOWCHROME_H

// ==============================================================
// WindowChrome（Qt Quick 版 UI 専用、Windows のみ有効）
//
//   OS のタイトルバーを消して、QML のヘッダーをタイトルバーとして使う。
//
//   Qt::FramelessWindowHint は使わない（スナップ・最大化アニメーション・影・
//   Win11 の角丸が消えるため）。ウィンドウスタイルは標準のまま、
//     WM_NCCALCSIZE : 上辺の非クライアント領域を 0 にする
//     WM_NCHITTEST  : 端 = リサイズ、ヘッダーの空き = キャプション（ドラッグ移動）
//   だけを差し替える。
//
//   titleBar に指定した Item の中で、マウスを受け取る子（ボタンなど）の上でなければ
//   キャプション扱い。QML 側でボタンを増やしてもここは直さなくてよい。
// ==============================================================

#include <QObject>
#include <QAbstractNativeEventFilter>
#include <QPointer>
#include <QQuickItem>

class QQuickWindow;

class WindowChrome : public QObject, public QAbstractNativeEventFilter
{
    Q_OBJECT
    // OS のタイトルバーを消せているか（false なら QML は自前のボタンを出さない）
    Q_PROPERTY(bool active READ active CONSTANT)
    // ドラッグで移動できるヘッダー
    Q_PROPERTY(QQuickItem* titleBar READ titleBar WRITE setTitleBar NOTIFY titleBarChanged)

public:
    explicit WindowChrome(QObject* parent = nullptr);
    ~WindowChrome() override;

    // QML のウィンドウができた後に 1 回呼ぶ
    void attach(QQuickWindow* window);

    bool active() const;
    QQuickItem* titleBar() const { return titleBar_; }
    void setTitleBar(QQuickItem* item);

    bool nativeEventFilter(const QByteArray& eventType, void* message, qintptr* result) override;

signals:
    void titleBarChanged();

private:
    // 物理ピクセル（クライアント座標）→ ヘッダー上の操作部品の上か
    bool isOverInteractive(QQuickItem* item, const QPointF& scenePos) const;
    long hitTest(long screenX, long screenY) const;

    QPointer<QQuickWindow> window_;
    QPointer<QQuickItem>   titleBar_;
    void* hwnd_ = nullptr;
};

#endif // WINDOWCHROME_H
