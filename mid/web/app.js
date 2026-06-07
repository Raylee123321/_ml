/* ==========================================================================
   F1 AI Telemetry Analysis Web Controller (Vanilla JS)
   Handles video play synchronisation, dynamic Chart.js drawing, and event jumping.
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    let telemetryData = null;
    let telemetryChart = null;
    let microChart = null;
    let microChartRange = null;
    let currentIdx = 0;
    let syncOffset = 0.0; // 影片與遙測數據的延遲修正 (秒)

    const video = document.getElementById('onboard-video');
    const offsetInput = document.getElementById('offset-input');
    const changeVideoBtn = document.getElementById('change-video-btn');
    const videoInput = document.getElementById('video-input');
    const dropZone = document.getElementById('drop-zone');

    // 1. 載入導出的 JSON 數據
    fetch('telemetry_data.json')
        .then(response => {
            if (!response.ok) {
                throw new Error("無法讀取 telemetry_data.json，請確保 export_web_data.py 已成功運行！");
            }
            return response.json();
        })
        .then(data => {
            telemetryData = data;
            initializeUI();
            initializeChart();
        })
        .catch(err => {
            console.error("載入遙測失敗:", err);
            document.getElementById('session-info').innerText = "載入遙測失敗: " + err.message;
        });

    // 2. 初始化 UI 面板與事件列表
    function initializeUI() {
        const info = telemetryData.info;
        document.getElementById('session-info').innerText = info.race_info;
        document.getElementById('driver-a-name').innerText = `${info.driver_a} (A)`;
        document.getElementById('driver-b-name').innerText = `${info.driver_b} (B)`;

        // 渲染 AI 診斷事件卡片
        const eventsList = document.getElementById('events-list');
        eventsList.innerHTML = ''; // 清除 loading 占位

        if (telemetryData.events.length === 0) {
            eventsList.innerHTML = '<div class="loading-placeholder">AI 未在此圈掃描到顯著行為差異。</div>';
            return;
        }

        telemetryData.events.forEach(event => {
            const card = document.createElement('div');
            card.className = `event-card ${event.type}`;
            card.id = `card-${event.id}`;
            
            // 點擊卡片跳轉影片播放時間 (提前 2.5 秒) 並展開微觀遙測圖表
            card.addEventListener('click', () => {
                jumpToEvent(event.key_playtime);
                showMicroTelemetry(event);
            });

            const badgeText = event.type === 'bottleneck' 
                ? `+${event.time_loss.toFixed(3)}s 損失`
                : `-${event.time_loss.toFixed(3)}s 領先`;

            card.innerHTML = `
                <div class="event-card-header">
                    <span class="event-title">${event.class_name_zh}</span>
                    <span class="event-metric-badge">${badgeText}</span>
                </div>
                <p class="event-desc">${event.description}</p>
                <div class="event-footer">
                    <span class="event-dist">📍 ${event.corner || '彎道'} (${Math.round(event.key_dist)}m)</span>
                    <button class="btn-jump">🔍 跳轉觀看</button>
                </div>
            `;
            eventsList.appendChild(card);
        });
    }

    // 3. 一鍵跳轉至瓶頸發生點
    function jumpToEvent(playtime) {
        // 設定播放時間（加入用戶修正的 syncOffset），並提早 2.5 秒以留出反應時間
        let targetTime = playtime - syncOffset - 2.5;
        if (targetTime < 0) targetTime = 0;
        
        video.currentTime = targetTime;
        video.play();
    }

    // 方案 B：動態生成微觀數據曲線圖 (前後 100m)
    function showMicroTelemetry(event) {
        document.querySelector('.video-card').classList.add('split-mode');
        
        const keyDist = event.key_dist;
        const startDist = Math.max(0, keyDist - 100);
        const endDist = Math.min(telemetryData.info.max_distance, keyDist + 100);
        microChartRange = { start: startDist, end: endDist, key: keyDist };
        
        const tel = telemetryData.telemetry;
        const indices = [];
        for (let i = 0; i < tel.distance.length; i++) {
            if (tel.distance[i] >= startDist && tel.distance[i] <= endDist) {
                indices.push(i);
            }
        }
        
        const mDistance = indices.map(i => tel.distance[i]);
        const mSpeedA = indices.map(i => tel.speed_a[i]);
        const mSpeedB = indices.map(i => tel.speed_b[i]);
        const mThrottleA = indices.map(i => tel.throttle_a[i]);
        const mThrottleB = indices.map(i => tel.throttle_b[i]);
        const mBrakeA = indices.map(i => tel.brake_a[i]);
        const mBrakeB = indices.map(i => tel.brake_b[i]);
        
        if (microChart) {
            microChart.destroy();
        }
        
        const ctx = document.getElementById('microTelemetryChart').getContext('2d');
        
        const keyPointPlugin = {
            id: 'keyPointIndicator',
            afterDraw: (chart) => {
                const ctx = chart.ctx;
                const xAxis = chart.scales.x;
                const yAxis = chart.scales.y_speed;
                if (!yAxis) return;
                
                const xPixel = xAxis.getPixelForValue(keyDist);
                
                ctx.save();
                ctx.beginPath();
                ctx.strokeStyle = '#ffd700'; // 亮金黃色
                ctx.lineWidth = 2;
                ctx.moveTo(xPixel, chart.chartArea.top);
                ctx.lineTo(xPixel, chart.chartArea.bottom);
                ctx.stroke();
                
                ctx.fillStyle = '#ffd700';
                ctx.font = 'bold 10px Outfit';
                ctx.fillText('AI KEY POINT', xPixel + 5, chart.chartArea.top + 15);
                ctx.restore();
            }
        };
        
        const microCursorPlugin = {
            id: 'microCursor',
            afterDraw: (chart) => {
                if (currentIdx === null || !telemetryData) return;
                const currentDist = telemetryData.telemetry.distance[currentIdx];
                if (currentDist < startDist || currentDist > endDist) return;
                
                const ctx = chart.ctx;
                const xAxis = chart.scales.x;
                const xPixel = xAxis.getPixelForValue(currentDist);
                
                ctx.save();
                ctx.beginPath();
                ctx.strokeStyle = '#e10600';
                ctx.lineWidth = 1.5;
                ctx.setLineDash([3, 3]);
                ctx.moveTo(xPixel, chart.chartArea.top);
                ctx.lineTo(xPixel, chart.chartArea.bottom);
                ctx.stroke();
                ctx.restore();
            }
        };
        
        microChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: mDistance,
                datasets: [
                    {
                        label: 'Speed A',
                        data: mSpeedA,
                        borderColor: '#00d2c4',
                        borderWidth: 2.5,
                        pointRadius: 0,
                        yAxisID: 'y_speed',
                        tension: 0.1
                    },
                    {
                        label: 'Speed B',
                        data: mSpeedB,
                        borderColor: '#e10600',
                        borderWidth: 2.5,
                        pointRadius: 0,
                        yAxisID: 'y_speed',
                        tension: 0.1
                    },
                    {
                        label: 'Throttle A',
                        data: mThrottleA,
                        borderColor: 'rgba(57, 255, 20, 0.85)',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        yAxisID: 'y_pedals',
                        tension: 0.1
                    },
                    {
                        label: 'Throttle B',
                        data: mThrottleB,
                        borderColor: 'rgba(57, 255, 20, 0.35)',
                        borderWidth: 1.5,
                        borderDash: [3, 3],
                        pointRadius: 0,
                        yAxisID: 'y_pedals',
                        tension: 0.1
                    },
                    {
                        label: 'Brake A',
                        data: mBrakeA,
                        borderColor: 'rgba(255, 7, 58, 0.85)',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        yAxisID: 'y_pedals',
                        tension: 0.1
                    },
                    {
                        label: 'Brake B',
                        data: mBrakeB,
                        borderColor: 'rgba(255, 7, 58, 0.35)',
                        borderWidth: 1.5,
                        borderDash: [3, 3],
                        pointRadius: 0,
                        yAxisID: 'y_pedals',
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                hover: { mode: null },
                scales: {
                    x: {
                        type: 'linear',
                        min: startDist,
                        max: endDist,
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#90909c', font: { size: 9 } }
                    },
                    y_speed: {
                        type: 'linear',
                        position: 'left',
                        title: { display: false },
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#90909c', font: { size: 9 } }
                    },
                    y_pedals: {
                        type: 'linear',
                        position: 'right',
                        min: 0,
                        max: 100,
                        title: { display: false },
                        grid: { drawOnChartArea: false },
                        ticks: { color: '#90909c', font: { size: 9 } }
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            },
            plugins: [keyPointPlugin, microCursorPlugin]
        });
    }

    // 4. 初始化 Chart.js 遙測對照圖表
    function initializeChart() {
        const ctx = document.getElementById('telemetryChart').getContext('2d');
        const tel = telemetryData.telemetry;
        
        // 為了網頁流暢度與渲染速度，我們可以使用抽樣（Downsampling），但 1500 個點 Chart.js 能輕鬆應對
        const labels = tel.distance;

        // 背景高亮插件：在圖表背景繪製 AI 偵測到的失誤彎道區域
        const backgroundHighlightPlugin = {
            id: 'backgroundHighlight',
            beforeDraw: (chart) => {
                const ctx = chart.ctx;
                const xAxis = chart.scales.x;
                
                telemetryData.events.forEach(event => {
                    const startX = xAxis.getPixelForValue(event.start_dist);
                    const endX = xAxis.getPixelForValue(event.end_dist);
                    const height = chart.chartArea.bottom - chart.chartArea.top;
                    
                    ctx.save();
                    // 瓶頸用極淡紅，優勢用極淡綠
                    ctx.fillStyle = event.type === 'bottleneck' 
                        ? 'rgba(225, 6, 0, 0.06)' 
                        : 'rgba(0, 210, 196, 0.06)';
                    ctx.fillRect(startX, chart.chartArea.top, endX - startX, height);
                    ctx.restore();
                });
            }
        };

        // 垂直游標指示線插件：影片播放時繪製虛線對齊當前距離點
        const verticalCursorPlugin = {
            id: 'verticalCursor',
            afterDraw: (chart) => {
                if (currentIdx === null || !telemetryData) return;
                const ctx = chart.ctx;
                const xAxis = chart.scales.x;
                const xVal = telemetryData.telemetry.distance[currentIdx];
                const xPixel = xAxis.getPixelForValue(xVal);
                
                ctx.save();
                ctx.beginPath();
                ctx.strokeStyle = '#e10600';
                ctx.lineWidth = 1.5;
                ctx.setLineDash([4, 4]); // 虛線效果
                ctx.moveTo(xPixel, chart.chartArea.top);
                ctx.lineTo(xPixel, chart.chartArea.bottom);
                ctx.stroke();
                ctx.restore();
            }
        };

        telemetryChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    // 速度 A (ANT - 青色)
                    {
                        label: `Speed ${telemetryData.info.driver_a}`,
                        data: tel.speed_a,
                        borderColor: '#00d2c4',
                        borderWidth: 2,
                        pointRadius: 0,
                        yAxisID: 'ySpeed',
                        tension: 0.1
                    },
                    // 速度 B (VER - 紅色)
                    {
                        label: `Speed ${telemetryData.info.driver_b}`,
                        data: tel.speed_b,
                        borderColor: '#e10600',
                        borderWidth: 2,
                        pointRadius: 0,
                        yAxisID: 'ySpeed',
                        tension: 0.1
                    },
                    // 油門 A (ANT - 淡綠細線)
                    {
                        label: `Throttle A`,
                        data: tel.throttle_a,
                        borderColor: 'rgba(57, 255, 20, 0.3)',
                        borderWidth: 1,
                        pointRadius: 0,
                        yAxisID: 'yPedals',
                        tension: 0.1
                    },
                    // 煞車 A (ANT - 淡紅細線)
                    {
                        label: `Brake A`,
                        data: tel.brake_a,
                        borderColor: 'rgba(255, 7, 58, 0.3)',
                        borderWidth: 1,
                        pointRadius: 0,
                        yAxisID: 'yPedals',
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false, // 關閉圖表動畫，在 video 播放的高頻渲染中確保極高流暢性
                hover: { mode: null },
                tooltips: { enabled: false },
                scales: {
                    x: {
                        type: 'linear',
                        title: { display: true, text: 'Distance (m)', color: '#90909c' },
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#90909c' }
                    },
                    ySpeed: {
                        type: 'linear',
                        position: 'left',
                        min: 30,
                        max: 340,
                        title: { display: true, text: 'Speed (km/h)', color: '#00d2c4' },
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#90909c' }
                    },
                    yPedals: {
                        type: 'linear',
                        position: 'right',
                        min: 0,
                        max: 100,
                        title: { display: true, text: 'Pedal (%)', color: 'rgba(57, 255, 20, 0.6)' },
                        grid: { drawOnChartArea: false }, // 避免與速度網格衝突
                        ticks: { color: '#90909c' }
                    }
                },
                plugins: {
                    legend: { display: false } // 使用自訂精美 legend
                }
            },
            plugins: [backgroundHighlightPlugin, verticalCursorPlugin]
        });
    }

    // 5. 影片播放時間與遙測距離對齊
    video.addEventListener('timeupdate', () => {
        if (!telemetryData) return;

        const playtimeArr = telemetryData.telemetry.playtime_a;
        // 影片目前對應的遙測 PlayTime
        const currentPlaytime = video.currentTime + syncOffset;

        // 在 playtime 數組中，利用二分搜尋法快速定位最接近的點 (因為 playtime 單調遞增)
        let idx = binarySearchClosest(playtimeArr, currentPlaytime);
        
        if (idx !== currentIdx) {
            currentIdx = idx;
            updateDashboardMetrics(idx);
            highlightActiveEvent(idx);
            
            // 刷新圖表垂直引導線
            if (telemetryChart) {
                telemetryChart.update('none'); // 用 'none' 模式不觸發 redraw 動畫，流暢度滿分
            }
            if (microChart && microChartRange) {
                const currentDist = telemetryData.telemetry.distance[idx];
                if (currentDist >= microChartRange.start && currentDist <= microChartRange.end) {
                    microChart.update('none');
                }
            }
        }
    });

    // 6. 二分搜尋法獲取最接近時間的索引
    function binarySearchClosest(arr, target) {
        let left = 0;
        let right = arr.length - 1;
        
        if (target <= arr[left]) return left;
        if (target >= arr[right]) return right;
        
        while (left <= right) {
            let mid = Math.floor((left + right) / 2);
            if (arr[mid] === target) return mid;
            
            if (target < arr[mid]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        // 比對最靠近的兩個值
        return (arr[left] - target) < (target - arr[right]) ? left : right;
    }

    // 7. 更新儀表板數據與踏板指示條
    function updateDashboardMetrics(idx) {
        const tel = telemetryData.telemetry;
        
        const speedA = tel.speed_a[idx];
        const speedB = tel.speed_b[idx];
        const throttleA = tel.throttle_a[idx];
        const throttleB = tel.throttle_b[idx];
        const brakeA = tel.brake_a[idx];
        const brakeB = tel.brake_b[idx];
        const distance = tel.distance[idx];
        const delta = tel.playtime_a[idx] - tel.playtime_b[idx];

        // 數位速度更新
        document.getElementById('speed-a-val').innerText = Math.round(speedA);
        document.getElementById('speed-b-val').innerText = Math.round(speedB);

        // 踏板數值與進度條
        document.getElementById('throttle-a-bar').style.width = `${throttleA}%`;
        document.getElementById('throttle-a-txt').innerText = `${Math.round(throttleA)}%`;
        document.getElementById('brake-a-bar').style.width = `${brakeA}%`;
        document.getElementById('brake-a-txt').innerText = `${Math.round(brakeA)}%`;

        document.getElementById('throttle-b-bar').style.width = `${throttleB}%`;
        document.getElementById('throttle-b-txt').innerText = `${Math.round(throttleB)}%`;
        document.getElementById('brake-b-bar').style.width = `${brakeB}%`;
        document.getElementById('brake-b-txt').innerText = `${Math.round(brakeB)}%`;

        // 距離點更新
        document.getElementById('live-dist-val').innerText = `${Math.round(distance)}m`;

        // 即時差值更新 (領先為負，落後為正)
        const deltaEl = document.getElementById('live-delta-val');
        if (delta > 0.005) {
            deltaEl.innerText = `+${delta.toFixed(3)}s`;
            deltaEl.className = "delta-value behind";
        } else if (delta < -0.005) {
            deltaEl.innerText = `${delta.toFixed(3)}s`;
            deltaEl.className = "delta-value ahead";
        } else {
            deltaEl.innerText = "0.000s";
            deltaEl.className = "delta-value neutral";
        }
    }

    // 8. 當行駛到特定失誤區段時，在右側面板高亮對應卡片
    function highlightActiveEvent(idx) {
        if (!telemetryData) return;
        const currentDist = telemetryData.telemetry.distance[idx];
        
        telemetryData.events.forEach(event => {
            const card = document.getElementById(`card-${event.id}`);
            if (!card) return;
            
            // 若當前位置在該事件的起終點區間內，標記為 active
            if (currentDist >= event.start_dist && currentDist <= event.end_dist) {
                card.classList.add('active-playing');
                // 自動將 active 卡片滾動到視野中
                card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            } else {
                card.classList.remove('active-playing');
            }
        });
    }

    // 9. 監聽修正延遲輸入
    offsetInput.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        if (!isNaN(val)) {
            syncOffset = val;
        }
    });

    // 10. 拖曳上傳/更換本地 Onboard 影片 (Drag & Drop)
    changeVideoBtn.addEventListener('click', () => {
        videoInput.click();
    });

    videoInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            loadLocalVideo(file);
        }
    });

    // 拖曳事件處理
    window.addEventListener('dragenter', (e) => {
        e.preventDefault();
        dropZone.classList.add('active');
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        // 僅當離開 dropZone 本身時關閉 overlay
        if (e.relatedTarget === null || !dropZone.contains(e.relatedTarget)) {
            dropZone.classList.remove('active');
        }
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('active');
        const file = e.dataTransfer.files[0];
        if (file && file.type === 'video/mp4') {
            loadLocalVideo(file);
        } else {
            alert("請上傳 MP4 格式的 F1 單圈影片！");
        }
    });

    function loadLocalVideo(file) {
        const fileURL = URL.createObjectURL(file);
        video.src = fileURL;
        video.load();
        video.play();
        console.log(`成功載入本地 onboard 影片: ${file.name}`);
    }
});

// CSS 追加 active 卡片樣式 (在 JS 動態插入或直接寫在 CSS 中)
const style = document.createElement('style');
style.innerHTML = `
    .event-card.active-playing {
        border-color: #ffd700 !important;
        box-shadow: 0 0 15px rgba(255, 215, 0, 0.25) !important;
        transform: scale(1.02);
    }
`;
document.head.appendChild(style);
