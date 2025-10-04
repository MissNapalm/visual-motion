<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gesture Browser</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            overflow: hidden;
            height: 100vh;
            background: #1a1a1a;
        }

        .browser-chrome {
            background: #f0f0f0;
            border-bottom: 1px solid #ccc;
            display: flex;
            flex-direction: column;
            -webkit-app-region: drag;
        }

        .window-controls {
            display: flex;
            gap: 8px;
            padding: 12px;
            -webkit-app-region: no-drag;
        }

        .control {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }

        .close { background: #ff5f57; }
        .minimize { background: #ffbd2e; }
        .maximize { background: #28ca42; }

        .navigation {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            -webkit-app-region: no-drag;
        }

        .nav-btn {
            width: 32px;
            height: 32px;
            border: none;
            background: #e0e0e0;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            transition: background 0.2s;
        }

        .nav-btn:hover {
            background: #d0d0d0;
        }

        .address-bar-container {
            flex: 1;
        }

        .address-bar {
            width: 100%;
            padding: 8px 16px;
            border: 1px solid #ccc;
            border-radius: 20px;
            font-size: 14px;
            outline: none;
        }

        .address-bar:focus {
            border-color: #4285f4;
        }

        .browser-content {
            flex: 1;
            height: calc(100vh - 100px);
            position: relative;
            z-index: 1;
        }

        webview {
            width: 100%;
            height: 100%;
            border: none;
            position: relative;
            z-index: 1;
        }

        .camera-overlay {
            position: fixed;
            top: 100px;
            right: 20px;
            width: 320px;
            height: 240px;
            background: rgba(0, 0, 0, 0.9);
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            z-index: 1000;
        }

        .camera-header {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 8px 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 12px;
            z-index: 1001;
        }

        #toggleCamera {
            background: none;
            border: none;
            cursor: pointer;
            font-size: 16px;
            color: white;
        }

        #video {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transform: scaleX(-1);
        }

        #canvas {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            transform: scaleX(-1);
        }

        .gesture-indicator {
            position: fixed;
            bottom: 20px;
            left: 20px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            font-size: 14px;
            z-index: 1000;
        }

        .pinch-dot {
            position: fixed;
            width: 24px;
            height: 24px;
            border: 3px solid #4285f4;
            border-radius: 50%;
            background: rgba(66, 133, 244, 0.3);
            transform: translate(-50%, -50%);
            pointer-events: none;
            z-index: 2100;
            display: none;
            box-shadow: 0 0 20px rgba(66, 133, 244, 0.6);
            transition: all 0.1s ease;
        }

        .pinch-dot.active {
            display: block;
        }
        
        .pinch-dot.pinching {
            animation: pulse 0.6s ease-in-out infinite;
            border-color: #00ff00;
            background: rgba(0, 255, 0, 0.4);
            box-shadow: 0 0 30px rgba(0, 255, 0, 0.8);
        }

        @keyframes pulse {
            0%, 100% { transform: translate(-50%, -50%) scale(1); }
            50% { transform: translate(-50%, -50%) scale(1.2); }
        }

        /* Full-screen hand overlay */
        #handOverlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 999999;
            background: transparent;
        }

        /* On-screen keyboard */
        .on-screen-keyboard {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(30, 30, 30, 0.95);
            padding: 20px;
            z-index: 2000;
            box-shadow: 0 -4px 20px rgba(0, 0, 0, 0.5);
        }

        .keyboard-row {
            display: flex;
            justify-content: center;
            gap: 8px;
            margin-bottom: 8px;
        }

        .keyboard-key {
            min-width: 60px;
            height: 50px;
            background: #4a4a4a;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            cursor: pointer;
            transition: all 0.1s;
        }

        .keyboard-key:hover {
            background: #5a5a5a;
            transform: scale(1.05);
        }

        .keyboard-key:active {
            background: #3a3a3a;
            transform: scale(0.95);
        }

        .key-space {
            min-width: 200px;
        }

        .key-action {
            background: #4285f4;
            min-width: 100px;
        }

        .key-action:hover {
            background: #5295ff;
        }
    </style>
</head>
<body>
    <!-- Browser Chrome -->
    <div class="browser-chrome">
        <div class="window-controls">
            <div class="control close"></div>
            <div class="control minimize"></div>
            <div class="control maximize"></div>
        </div>
        
        <div class="navigation">
            <button class="nav-btn" id="backBtn">←</button>
            <button class="nav-btn" id="forwardBtn">→</button>
            <button class="nav-btn" id="reloadBtn">↻</button>
            
            <div class="address-bar-container">
                <input 
                    type="text" 
                    id="addressBar" 
                    class="address-bar" 
                    placeholder="Enter URL or search..."
                    value="https://www.youtube.com"
                >
            </div>
            
            <button class="nav-btn" id="gestureToggle">🤚</button>
        </div>
    </div>

    <!-- Browser Content (WebView) -->
    <div class="browser-content">
        <webview 
            id="webview" 
            src="https://www.youtube.com"
            allowpopups
        ></webview>
        <!-- Hand overlay canvas INSIDE browser-content -->
        <canvas id="handOverlay"></canvas>
    </div>

    <!-- Camera Overlay -->
    <div class="camera-overlay" id="cameraOverlay">
        <div class="camera-header">
            <span>Hand Tracking</span>
            <button id="toggleCamera">📷</button>
        </div>
        <video id="video" autoplay playsinline></video>
        <canvas id="canvas"></canvas>
    </div>

    <!-- Gesture Indicator -->
    <div class="gesture-indicator" id="gestureIndicator">
        Initializing...
    </div>

    <!-- Pinch Visualizers -->
    <div class="pinch-dot" id="pinch1"></div>
    <div class="pinch-dot" id="pinch2"></div>

    <!-- MediaPipe -->
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js"></script>
    
    <script>
        // Import MediaPipe constants
        const { HAND_CONNECTIONS } = window;
        
        // ============================================
        // GESTURE CONTROLLER
        // ============================================
        class GestureController {
            constructor() {
                this.smoothingFactor = 0.7; // Increased from 0.3 for faster tracking
                this.prevPositions = new Map();
            }

            smooth(key, value) {
                if (!this.prevPositions.has(key)) {
                    this.prevPositions.set(key, value);
                    return value;
                }
                
                const prev = this.prevPositions.get(key);
                const smoothed = this.smoothingFactor * value + (1 - this.smoothingFactor) * prev;
                this.prevPositions.set(key, smoothed);
                return smoothed;
            }

            distance(p1, p2) {
                return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
            }

            isPinching(landmarks) {
                const thumbTip = landmarks[4];
                const indexTip = landmarks[8];
                return this.distance(thumbTip, indexTip) < 0.05;
            }

            getPinchPosition(landmarks) {
                const thumbTip = landmarks[4];
                const indexTip = landmarks[8];
                return {
                    x: (thumbTip.x + indexTip.x) / 2,
                    y: (thumbTip.y + indexTip.y) / 2
                };
            }

            detectSwipe(landmarks) {
                const wrist = landmarks[0];
                const middleFinger = landmarks[12];
                const dx = middleFinger.x - wrist.x;
                
                if (Math.abs(dx) > 0.3) {
                    return dx > 0 ? 'right' : 'left';
                }
                return null;
            }

            isPointingUp(landmarks) {
                // Index finger tip (8) should be above index base (5)
                // and other fingers should be curled down
                const indexTip = landmarks[8];
                const indexBase = landmarks[5];
                const middleTip = landmarks[12];
                const ringTip = landmarks[16];
                const pinkyTip = landmarks[20];
                const wrist = landmarks[0];
                
                // Index finger pointing up
                const indexUp = indexTip.y < indexBase.y - 0.1;
                
                // Other fingers curled (tips not much higher than base)
                const othersCurled = 
                    middleTip.y > wrist.y - 0.1 &&
                    ringTip.y > wrist.y - 0.1 &&
                    pinkyTip.y > wrist.y - 0.1;
                
                return indexUp && othersCurled;
            }
        }

        // ============================================
        // BROWSER CONTROLLER
        // ============================================
        class BrowserController {
            constructor() {
                this.webview = document.getElementById('webview');
                this.addressBar = document.getElementById('addressBar');
                this.backBtn = document.getElementById('backBtn');
                this.forwardBtn = document.getElementById('forwardBtn');
                this.reloadBtn = document.getElementById('reloadBtn');
                
                this.scale = 1;
                this.lastPinchPos = null;
                this.lastPinchDist = null;
                this.overlayInjected = false;
                
                // Velocity tracking for momentum scrolling
                this.velocity = { x: 0, y: 0 };
                this.lastMoveTime = 0;
                this.scrollHistory = [];
                
                // Tab management
                this.tabs = [{ url: 'https://www.youtube.com', title: 'YouTube' }];
                this.currentTabIndex = 0;
                
                // Keyboard state
                this.keyboardVisible = false;
                this.lastKeyboardPinch = false;
                this.lastWebviewClick = false;
                
                this.setupEventListeners();
                this.createKeyboard();
            }

            setupEventListeners() {
                this.addressBar.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') {
                        this.navigate(this.addressBar.value);
                    }
                });

                this.backBtn.addEventListener('click', () => this.goBack());
                this.forwardBtn.addEventListener('click', () => this.goForward());
                this.reloadBtn.addEventListener('click', () => this.reload());

                this.webview.addEventListener('did-navigate', (e) => {
                    this.addressBar.value = e.url;
                    this.overlayInjected = false;
                });

                this.webview.addEventListener('did-navigate-in-page', (e) => {
                    this.addressBar.value = e.url;
                });

                // Inject overlay when page loads
                this.webview.addEventListener('dom-ready', () => {
                    console.log('🌐 Webview DOM ready, injecting overlay...');
                    setTimeout(() => {
                        this.injectHandOverlay();
                    }, 500); // Small delay to ensure page is ready
                });
            }

            injectHandOverlay() {
                if (this.overlayInjected) return;
                
                console.log('💉 Injecting hand overlay into webview...');
                
                // Inject canvas and drawing code into the webview
                this.webview.executeJavaScript(`
                    console.log('🔧 Starting injection...');
                    
                    // Remove old overlay if exists
                    const oldOverlay = document.getElementById('gesture-hand-overlay');
                    if (oldOverlay) oldOverlay.remove();
                    
                    // Create overlay canvas
                    const overlay = document.createElement('canvas');
                    overlay.id = 'gesture-hand-overlay';
                    overlay.style.position = 'fixed';
                    overlay.style.top = '0';
                    overlay.style.left = '0';
                    overlay.style.width = '100vw';
                    overlay.style.height = '100vh';
                    overlay.style.pointerEvents = 'none';
                    overlay.style.zIndex = '999999';
                    overlay.width = window.innerWidth;
                    overlay.height = window.innerHeight;
                    document.body.appendChild(overlay);
                    
                    console.log('📐 Overlay canvas created:', overlay.width, 'x', overlay.height);
                    
                    const ctx = overlay.getContext('2d');
                    
                    // Resize handler
                    window.addEventListener('resize', () => {
                        overlay.width = window.innerWidth;
                        overlay.height = window.innerHeight;
                    });
                    
                    // Drawing function called from parent
                    window.drawHands = function(handsData) {
                        ctx.clearRect(0, 0, overlay.width, overlay.height);
                        
                        if (!handsData || handsData.length === 0) {
                            return;
                        }
                        
                        handsData.forEach((hand, idx) => {
                            // Draw connections
                            ctx.strokeStyle = '#0088FF';
                            ctx.lineWidth = 4;
                            hand.connections.forEach(conn => {
                                ctx.beginPath();
                                ctx.moveTo(conn.start.x, conn.start.y);
                                ctx.lineTo(conn.end.x, conn.end.y);
                                ctx.stroke();
                            });
                            
                            // Draw landmarks
                            hand.landmarks.forEach(point => {
                                ctx.fillStyle = '#00DDFF';
                                ctx.beginPath();
                                ctx.arc(point.x, point.y, 8, 0, 2 * Math.PI);
                                ctx.fill();
                                
                                ctx.strokeStyle = '#FFFFFF';
                                ctx.lineWidth = 2;
                                ctx.stroke();
                            });
                        });
                    };
                    
                    console.log('✅ Hand overlay injected and ready!');
                `).then(() => {
                    console.log('✅ Injection complete');
                    this.overlayInjected = true;
                }).catch(err => {
                    console.error('❌ Injection failed:', err);
                });
            }

            drawHandsInWebview(handsData) {
                if (!this.overlayInjected) {
                    console.log('⚠️ Overlay not injected yet, skipping draw');
                    return;
                }
                
                console.log('🎨 Drawing', handsData.length, 'hands in webview');
                
                // Call the drawHands function inside the webview
                this.webview.executeJavaScript(`
                    if (window.drawHands) {
                        window.drawHands(${JSON.stringify(handsData)});
                    } else {
                        console.error('drawHands function not found!');
                    }
                `).catch(err => {
                    console.error('Error drawing in webview:', err);
                });
            }

            navigate(url) {
                if (!url.startsWith('http://') && !url.startsWith('https://')) {
                    url = 'https://' + url;
                }
                this.webview.src = url;
            }

            goBack() {
                if (this.webview.canGoBack()) {
                    this.webview.goBack();
                }
            }

            goForward() {
                if (this.webview.canGoForward()) {
                    this.webview.goForward();
                }
            }

            reload() {
                this.webview.reload();
            }

            handleSinglePinch(position) {
                const rect = this.webview.getBoundingClientRect();
                const x = position.x * rect.width;
                const y = position.y * rect.height;
                const now = Date.now();

                if (this.lastPinchPos) {
                    const dx = x - this.lastPinchPos.x;
                    const dy = y - this.lastPinchPos.y;
                    const dt = now - this.lastMoveTime;
                    
                    // Calculate velocity (pixels per ms)
                    if (dt > 0) {
                        const vx = dx / dt;
                        const vy = dy / dt;
                        
                        // Keep scroll history for momentum calculation
                        this.scrollHistory.push({ vx, vy, time: now });
                        
                        // Keep only recent history (last 100ms)
                        this.scrollHistory = this.scrollHistory.filter(h => now - h.time < 100);
                    }
                    
                    // Immediate scroll response with amplification
                    this.webview.executeJavaScript(`
                        window.scrollBy(${-dx * 3}, ${-dy * 3});
                    `);
                }

                this.lastPinchPos = { x, y };
                this.lastMoveTime = now;
            }

            handlePinchRelease() {
                // Calculate average velocity from recent history
                if (this.scrollHistory.length > 0) {
                    const avgVx = this.scrollHistory.reduce((sum, h) => sum + h.vx, 0) / this.scrollHistory.length;
                    const avgVy = this.scrollHistory.reduce((sum, h) => sum + h.vy, 0) / this.scrollHistory.length;
                    
                    // Apply momentum scroll with deceleration
                    const momentumX = Math.round(-avgVx * 800);
                    const momentumY = Math.round(-avgVy * 800);
                    
                    // Only apply momentum if there's significant velocity
                    if (Math.abs(momentumX) > 10 || Math.abs(momentumY) > 10) {
                        this.webview.executeJavaScript(`
                            (function() {
                                const startScroll = { x: window.scrollX, y: window.scrollY };
                                const targetScroll = { 
                                    x: window.scrollX + ${momentumX}, 
                                    y: window.scrollY + ${momentumY} 
                                };
                                
                                let startTime = null;
                                const duration = 600;
                                
                                function easeOutCubic(t) {
                                    return 1 - Math.pow(1 - t, 3);
                                }
                                
                                function animate(currentTime) {
                                    if (!startTime) startTime = currentTime;
                                    const elapsed = currentTime - startTime;
                                    const progress = Math.min(elapsed / duration, 1);
                                    const eased = easeOutCubic(progress);
                                    
                                    window.scrollTo(
                                        startScroll.x + (targetScroll.x - startScroll.x) * eased,
                                        startScroll.y + (targetScroll.y - startScroll.y) * eased
                                    );
                                    
                                    if (progress < 1) {
                                        requestAnimationFrame(animate);
                                    }
                                }
                                
                                requestAnimationFrame(animate);
                            })();
                        `).catch(err => {
                            console.log('Momentum scroll error (page may not be ready):', err.message);
                        });
                    }
                }
                
                // Clear history
                this.scrollHistory = [];
            }

            handleTwoHandPinch(pos1, pos2) {
                const dist = Math.sqrt(
                    Math.pow(pos1.x - pos2.x, 2) + 
                    Math.pow(pos1.y - pos2.y, 2)
                );

                if (this.lastPinchDist) {
                    const delta = dist - this.lastPinchDist;
                    this.scale = Math.max(0.5, Math.min(3, this.scale + delta * 2));
                    this.webview.setZoomFactor(this.scale);
                }

                this.lastPinchDist = dist;
                this.lastPinchPos = null;
            }

            handleSwipe(direction) {
                if (direction === 'left') {
                    this.goForward();
                } else if (direction === 'right') {
                    this.goBack();
                }
            }

            resetGesture() {
                // Trigger momentum when pinch is released
                if (this.lastPinchPos !== null) {
                    this.handlePinchRelease();
                }
                
                this.lastPinchPos = null;
                this.lastPinchDist = null;
            }

            createKeyboard() {
                const keyboard = document.createElement('div');
                keyboard.id = 'onScreenKeyboard';
                keyboard.className = 'on-screen-keyboard';
                keyboard.style.display = 'none';
                
                const keys = [
                    ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
                    ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
                    ['z', 'x', 'c', 'v', 'b', 'n', 'm', '⌫'],
                    ['.com', 'space', 'Go']
                ];
                
                keys.forEach(row => {
                    const rowDiv = document.createElement('div');
                    rowDiv.className = 'keyboard-row';
                    
                    row.forEach(key => {
                        const keyBtn = document.createElement('button');
                        keyBtn.className = 'keyboard-key';
                        keyBtn.textContent = key === 'space' ? '___' : key;
                        keyBtn.dataset.key = key;
                        
                        if (key === 'Go') keyBtn.classList.add('key-action');
                        if (key === 'space') keyBtn.classList.add('key-space');
                        
                        keyBtn.addEventListener('click', () => this.handleKeyPress(key));
                        rowDiv.appendChild(keyBtn);
                    });
                    
                    keyboard.appendChild(rowDiv);
                });
                
                document.body.appendChild(keyboard);
                this.keyboard = keyboard;
            }

            handleKeyPress(key) {
                const addressBar = this.addressBar;
                
                if (key === '⌫') {
                    addressBar.value = addressBar.value.slice(0, -1);
                } else if (key === 'space') {
                    addressBar.value += ' ';
                } else if (key === 'Go') {
                    this.navigate(addressBar.value);
                    this.hideKeyboard();
                } else {
                    addressBar.value += key;
                }
                
                // Visual feedback
                const keyBtn = this.keyboard.querySelector(`[data-key="${key}"]`);
                if (keyBtn) {
                    keyBtn.style.background = '#2a2a2a';
                    setTimeout(() => {
                        keyBtn.style.background = key === 'Go' ? '#4285f4' : '#4a4a4a';
                    }, 100);
                }
            }

            checkPinchOnKeyboard(pinchX, pinchY, isPinching) {
                if (!this.keyboardVisible) return;
                
                // Only trigger on pinch start (not while holding)
                if (isPinching && !this.lastKeyboardPinch) {
                    console.log('🎯 Pinch at:', pinchX, pinchY);
                    const keys = this.keyboard.querySelectorAll('.keyboard-key');
                    let foundKey = false;
                    
                    keys.forEach(keyBtn => {
                        const rect = keyBtn.getBoundingClientRect();
                        
                        if (pinchX >= rect.left && pinchX <= rect.right &&
                            pinchY >= rect.top && pinchY <= rect.bottom) {
                            const key = keyBtn.dataset.key;
                            this.handleKeyPress(key);
                            console.log('⌨️ Typed:', key);
                            foundKey = true;
                        }
                    });
                    
                    if (!foundKey) {
                        console.log('❌ No key at pinch position');
                    }
                }
                
                this.lastKeyboardPinch = isPinching;
            }

            clickInWebview(x, y, isPinching) {
                // Only click on pinch start (not while holding)
                if (isPinching && !this.lastWebviewClick) {
                    // Convert to webview-local coordinates
                    const rect = this.webview.getBoundingClientRect();
                    const localX = x - rect.left;
                    const localY = y - rect.top;
                    
                    console.log('🖱️ Clicking at:', localX, localY);
                    
                    // Dispatch actual mouse events in the webview
                    this.webview.executeJavaScript(`
                        (function() {
                            const x = ${localX};
                            const y = ${localY};
                            const element = document.elementFromPoint(x, y);
                            
                            if (element) {
                                console.log('Element at point:', element.tagName, element.className, element.href);
                                
                                // Create and dispatch mouse events
                                const mousedown = new MouseEvent('mousedown', {
                                    bubbles: true,
                                    cancelable: true,
                                    view: window,
                                    clientX: x,
                                    clientY: y
                                });
                                
                                const mouseup = new MouseEvent('mouseup', {
                                    bubbles: true,
                                    cancelable: true,
                                    view: window,
                                    clientX: x,
                                    clientY: y
                                });
                                
                                const click = new MouseEvent('click', {
                                    bubbles: true,
                                    cancelable: true,
                                    view: window,
                                    clientX: x,
                                    clientY: y
                                });
                                
                                element.dispatchEvent(mousedown);
                                element.dispatchEvent(mouseup);
                                element.dispatchEvent(click);
                                
                                // Also try direct click
                                element.click();
                                
                                console.log('Click dispatched to', element);
                            } else {
                                console.log('No element at', x, y);
                            }
                        })();
                    `).catch(err => {
                        console.log('Click error:', err.message);
                    });
                }
                
                this.lastWebviewClick = isPinching;
            }

            showKeyboard() {
                if (!this.keyboardVisible) {
                    this.keyboard.style.display = 'block';
                    this.keyboardVisible = true;
                    this.addressBar.focus();
                }
            }

            hideKeyboard() {
                this.keyboard.style.display = 'none';
                this.keyboardVisible = false;
            }

            openNewTab() {
                this.tabs.push({ url: '', title: 'New Tab' });
                this.currentTabIndex = this.tabs.length - 1;
                this.webview.src = 'about:blank';
                this.addressBar.value = '';
                this.showKeyboard();
                console.log('📑 Opened new tab');
            }
        }

        // ============================================
        // MEDIAPIPE HANDLER
        // ============================================
        class MediaPipeHandler {
            constructor(onResults) {
                this.video = document.getElementById('video');
                this.canvas = document.getElementById('canvas');
                this.ctx = this.canvas.getContext('2d');
                this.onResultsCallback = onResults;
                this.frameCount = 0;
                
                this.initializeMediaPipe();
            }

            async initializeMediaPipe() {
                try {
                    console.log('Starting MediaPipe initialization...');
                    
                    const hands = new Hands({
                        locateFile: (file) => {
                            return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
                        }
                    });

                    hands.setOptions({
                        maxNumHands: 2,
                        modelComplexity: 1,
                        minDetectionConfidence: 0.5,  // Lowered from 0.7
                        minTrackingConfidence: 0.5    // Lowered from 0.7
                    });

                    hands.onResults((results) => {
                        console.log('⚡ Frame processed, hands:', results.multiHandLandmarks?.length || 0);
                        this.processResults(results);
                    });

                    console.log('Requesting camera access...');
                    const camera = new Camera(this.video, {
                        onFrame: async () => {
                            await hands.send({ image: this.video });
                        },
                        width: 640,
                        height: 480
                    });

                    await camera.start();
                    
                    // Wait for video to have actual dimensions
                    await new Promise((resolve) => {
                        const checkDimensions = () => {
                            if (this.video.videoWidth > 0 && this.video.videoHeight > 0) {
                                console.log('Video dimensions:', this.video.videoWidth, 'x', this.video.videoHeight);
                                resolve();
                            } else {
                                setTimeout(checkDimensions, 100);
                            }
                        };
                        checkDimensions();
                    });
                    
                    console.log('Camera started successfully!');
                    
                    // Check if video is actually playing
                    this.video.addEventListener('playing', () => {
                        console.log('Video is playing!');
                    });
                    
                    document.getElementById('gestureIndicator').textContent = 'Camera ready - show your hands!';
                } catch (error) {
                    console.error('MediaPipe initialization error:', error);
                    document.getElementById('gestureIndicator').textContent = 'Error: ' + error.message;
                }
            }

            processResults(results) {
                this.frameCount++;
                this.canvas.width = this.video.videoWidth;
                this.canvas.height = this.video.videoHeight;
                
                this.ctx.save();
                this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
                
                // Draw frame counter (proves canvas is updating)
                this.ctx.fillStyle = 'lime';
                this.ctx.font = 'bold 20px monospace';
                this.ctx.fillText(`Frame: ${this.frameCount}`, 10, 30);
                
                // Draw red box to show canvas bounds
                this.ctx.strokeStyle = 'red';
                this.ctx.lineWidth = 4;
                this.ctx.strokeRect(0, 0, this.canvas.width, this.canvas.height);
                
                if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
                    console.log('✋ DRAWING', results.multiHandLandmarks.length, 'hands!');
                    
                    for (const landmarks of results.multiHandLandmarks) {
                        // Draw with white lines and green dots
                        drawConnectors(this.ctx, landmarks, HAND_CONNECTIONS, 
                            { color: '#FFFFFF', lineWidth: 3 });
                        drawLandmarks(this.ctx, landmarks, 
                            { color: '#00FF00', lineWidth: 2, radius: 6 });
                    }
                } else {
                    // Draw "no hands" indicator
                    this.ctx.fillStyle = 'yellow';
                    this.ctx.font = '16px monospace';
                    this.ctx.fillText('No hands detected', 10, 60);
                }
                
                this.ctx.restore();
                
                if (this.onResultsCallback) {
                    this.onResultsCallback(results);
                }
            }
        }

        // ============================================
        // MAIN APPLICATION
        // ============================================
        const gestureController = new GestureController();
        const browserController = new BrowserController();
        const pinchDot1 = document.getElementById('pinch1');
        const pinchDot2 = document.getElementById('pinch2');
        const gestureIndicator = document.getElementById('gestureIndicator');
        const handOverlay = document.getElementById('handOverlay');
        const overlayCtx = handOverlay.getContext('2d');

        // Set overlay canvas to match browser content size
        function resizeOverlay() {
            const browserContent = document.querySelector('.browser-content');
            handOverlay.width = browserContent.offsetWidth;
            handOverlay.height = browserContent.offsetHeight;
        }
        resizeOverlay();
        window.addEventListener('resize', resizeOverlay);

        let lastSwipeTime = 0;
        let lastPointUpTime = 0;

        function onMediaPipeResults(results) {
            const pinches = [];
            
            // Clear the overlay canvas (for camera feed)
            overlayCtx.clearRect(0, 0, handOverlay.width, handOverlay.height);
            
            console.log('🖐️ Processing', results.multiHandLandmarks?.length || 0, 'hands');
            
            // Prepare hand data for webview injection
            const handsData = [];
            
            if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
                // Reset both indicators first
                let hand1Updated = false;
                let hand2Updated = false;
                
                results.multiHandLandmarks.forEach((landmarks, idx) => {
                    // Get webview dimensions (just width/height, not position)
                    const webview = document.getElementById('webview');
                    const rect = webview.getBoundingClientRect();
                    
                    // Get index finger tip position (landmark 8) - THIS is where we aim
                    const indexTip = landmarks[8];
                    const aimPosition = {
                        x: 1 - indexTip.x,  // No smoothing - instant tracking
                        y: indexTip.y       // No smoothing - instant tracking
                    };
                    
                    // Show aim indicator for this hand
                    if (idx === 0) {
                        updatePinchDot(pinchDot1, aimPosition, true, gestureController.isPinching(landmarks));
                        hand1Updated = true;
                    } else if (idx === 1) {
                        updatePinchDot(pinchDot2, aimPosition, true, gestureController.isPinching(landmarks));
                        hand2Updated = true;
                    }

                    // Check for pinching - use AIM position, not pinch midpoint
                    if (gestureController.isPinching(landmarks)) {
                        pinches.push({
                            x: aimPosition.x,  // Use index finger position
                            y: aimPosition.y,  // Use index finger position
                            idx: idx
                        });
                    }

                    // Check for swipe
                    const swipe = gestureController.detectSwipe(landmarks);
                    if (swipe && Date.now() - lastSwipeTime > 1000) {
                        browserController.handleSwipe(swipe);
                        lastSwipeTime = Date.now();
                        gestureIndicator.textContent = `Swipe ${swipe} detected`;
                    }
                });
                
                // Hide indicators for hands that aren't present
                if (!hand1Updated) {
                    updatePinchDot(pinchDot1, null, false, false);
                }
                if (!hand2Updated) {
                    updatePinchDot(pinchDot2, null, false, false);
                }
            } else {
                // No hands - hide all indicators
                updatePinchDot(pinchDot1, null, false, false);
                updatePinchDot(pinchDot2, null, false, false);
            }
            
            // Draw hands in the webview - DISABLED, only showing pinch indicators
            // console.log('📤 Sending', handsData.length, 'hands to webview');
            // browserController.drawHandsInWebview(handsData);
            
            // Handle pinch gestures
            if (pinches.length === 1) {
                const webview = document.getElementById('webview');
                const rect = webview.getBoundingClientRect();
                const pinchScreenX = rect.left + pinches[0].x * rect.width;
                const pinchScreenY = rect.top + pinches[0].y * rect.height;
                
                // Check if pinching on keyboard
                if (browserController.keyboardVisible) {
                    browserController.checkPinchOnKeyboard(pinchScreenX, pinchScreenY, true);
                } else {
                    // Click in webview
                    browserController.clickInWebview(pinchScreenX, pinchScreenY, true);
                    // Also handle scrolling
                    browserController.handleSinglePinch(pinches[0]);
                }
                
                gestureIndicator.textContent = '🤏 Pinching';
            } else if (pinches.length >= 2) {
                browserController.checkPinchOnKeyboard(0, 0, false);
                browserController.clickInWebview(0, 0, false);
                browserController.handleTwoHandPinch(pinches[0], pinches[1]);
                gestureIndicator.textContent = '🤏🤏 Zooming with two-hand pinch';
            } else {
                // Not pinching
                browserController.checkPinchOnKeyboard(0, 0, false);
                browserController.clickInWebview(0, 0, false);
                browserController.resetGesture();
                gestureIndicator.textContent = 'Aim with index finger';
            }
        }

        function updatePinchDot(dot, position, active, isPinching) {
            if (active && position) {
                // Get webview position on screen
                const webview = document.getElementById('webview');
                const rect = webview.getBoundingClientRect();
                
                // Position dot relative to webview position
                dot.style.left = (rect.left + position.x * rect.width) + 'px';
                dot.style.top = (rect.top + position.y * rect.height) + 'px';
                dot.classList.add('active');
                
                // Different style when pinching
                if (isPinching) {
                    dot.classList.add('pinching');
                } else {
                    dot.classList.remove('pinching');
                }
            } else {
                dot.classList.remove('active');
                dot.classList.remove('pinching');
            }
        }

        // Initialize MediaPipe
        const mediaPipeHandler = new MediaPipeHandler(onMediaPipeResults);

        // Camera toggle
        document.getElementById('toggleCamera').addEventListener('click', () => {
            const overlay = document.getElementById('cameraOverlay');
            overlay.style.display = overlay.style.display === 'none' ? 'block' : 'none';
        });

        console.log('Gesture Browser initialized!');
    </script>
</body>
</html>
