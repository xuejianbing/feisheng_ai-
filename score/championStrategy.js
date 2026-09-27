/*
 * ============================================
 * // 作者：飞升原创
 * 交流群: 123456789
 * v3.0
 * 版权所有，侵权必究
 * ============================================
 */

/* ================= 决策点嵌入固定（逐决策点 → 一条嵌入） =================
 * 训练后把每个"决策点"(action + id) 单独聚合成一条嵌入条，嵌入是该决策点
 * 全部样本特征向量的逐元素平均（与 FEATURE_DIM=130 对齐），同时记 TD 平均价值。
 * 运行中引擎以候选动作 (action,id) 精确命中对应嵌入条，按价值提权/替换。
 * 区别于旧版"每类只留一个冠军"，这里保留每个有效决策点一条嵌入，不再合并丢失。
 */

const STORE_KEY = 'djsc_champion_v1';
const VERSION = 2;               /* ★ v2：由"每类冠军"升级为"逐决策点嵌入条" */
const MIN_SAMPLES = 5;           /* 一个决策点至少几条样本才认为有效，防噪声 */
const MAX_PER_ACTION = 24;       /* 每类(action)最多保留的决策点嵌入条数，防存储膨胀 */

let STORE = load();

function load() {
    try {
        const raw = localStorage.getItem(STORE_KEY);
        if (raw) {
            const obj = JSON.parse(raw);
            if (obj && obj.v === VERSION && obj.embeds && typeof obj.embeds === 'object') {
                return { embeds: obj.embeds, updatedAt: obj.updatedAt || 0 };
            }
        }
    } catch (e) {}
    return { embeds: {}, updatedAt: 0 };
}

function persist() {
    try {
        STORE.updatedAt = Date.now();
        localStorage.setItem(STORE_KEY, JSON.stringify(STORE));
    } catch (e) {}
}

/* ★ 从样本重算"逐决策点嵌入条"：
 *  决策点 = (action, id)；对每个决策点求 f 的逐维平均（即嵌入，130 元素）+ TD 平均价值 + 样本数。
 *  group = { action: { id: { sumF:[130], sum, count, bestValue } } }
 *  每类只保留前 MAX_PER_ACTION 个 [样本数≥MIN_SAMPLES 且平均价值最高] 的决策点作为嵌入条。
 *  TD 用回填的 value_target（-1..1，越高越利于终局胜）。 */
export function recompute(samples) {
    try {
        const src = (samples && samples.length) ? samples : [];
        const group = {};
        for (let i = 0; i < src.length; i++) {
            const s = src[i];
            if (!s || !s.m) continue;
            const action = String(s.m.action || 'unknown');
            const id = String(s.m.id || '');
            if (!id) continue;
            const vt = (typeof s.value_target === 'number') ? s.value_target : 0;
            const w = (s.count || 1);
            const farr = (s.f && s.f.length) ? s.f : null;
            group[action] = group[action] || {};
            const g = group[action][id] || (group[action][id] = { sumF: null, hasF: 0, sum: 0, count: 0, bestValue: -Infinity });
            g.sum += vt * w;
            g.count += w;
            if (vt > g.bestValue) g.bestValue = vt;
            if (farr) {
                if (!g.sumF) g.sumF = new Array(farr.length).fill(0);
                for (let j = 0; j < farr.length; j++) g.sumF[j] += farr[j] * w;
                g.hasF += w;
            }
        }

        const embeds = {};
        for (const action in group) {
            if (!Object.prototype.hasOwnProperty.call(group, action)) continue;
            /* 取该类别所有有效决策点，按平均价值降序，仅保留前 MAX_PER_ACTION */
            const rows = [];
            for (const id in group[action]) {
                if (!Object.prototype.hasOwnProperty.call(group[action], id)) continue;
                const g = group[action][id];
                if (g.count < MIN_SAMPLES) continue;
                const avg = g.sum / g.count;
                if (avg <= 0) continue;   /* 仅保留价值为正（有决策意义） */
                let emb = null;
                if (g.sumF && g.hasF > 0) {
                    emb = new Array(g.sumF.length);
                    for (let j = 0; j < g.sumF.length; j++) emb[j] = Math.round((g.sumF[j] / g.hasF) * 10) / 10;
                }
                rows.push({ id: id, value: avg, count: g.count, bestValue: g.bestValue, emb: emb });
            }
            rows.sort(function (a, b) { return b.value - a.value; });
            if (rows.length) embeds[action] = rows.slice(0, MAX_PER_ACTION);
        }

        STORE.embeds = embeds;
        persist();
        return embeds;
    } catch (e) { return STORE.embeds || {}; }
}

/* ★ 读取当前固化的嵌入条： { action: [ {id,value,count,bestValue,emb} ] } */
export function getChampions() { return STORE.embeds || {}; }

/* ★ 查询某决策类型的"冠军"（即该类价值最高的一条嵌入），无则 null；兼容旧面板 */
export function championOf(action) {
    try {
        const rows = (STORE.embeds && STORE.embeds[action]) || [];
        return (rows.length ? rows[0] : null) || null;
    } catch (e) { return null; }
}

/* ★ 按 (action,id) 精确查某决策点的嵌入条，无则 null */
export function getEmbedding(action, id) {
    try {
        const rows = (STORE.embeds && STORE.embeds[action]) || [];
        for (let i = 0; i < rows.length; i++) {
            if (rows[i].id === id) return rows[i];
        }
    } catch (e) {}
    return null;
}

/* ★ 引擎调用：逐决策点嵌入替换规则链。
 * 命中判定用 (action,id) 精确命中对应嵌入条（用户设定：id 精确命中）。
 * 命中后按该决策点的 TD 价值把提权/替换强度缩放，价值越高越优先采用。
 *
 * 规则链（任一失败即回退，绝不破坏引擎）：
 *  R1 启用判定：boost>0 且已固化嵌入条，否则跳过。
 *  R2 类型锚定：只对同类型(action)的嵌入条生效，不同类互不干扰。
 *  R3 决策点命中：候选 acts 中存在 id 与嵌入条 id 完全一致的动作（精确命中）。
 *  R4 模糊判定：原最优 best 与次优分差 gap ≤ 阈值才允许替换（决策本就摇摆，风险低）。
 *  R5 价值校验：该决策点平均价值必须为正（value>0），否则排除。
 *  R6 提权：对命中的动作加 boost ×（0.5 + value），价值越高提权越多。
 *  R7 候选替换：强化后该动作成为最优且 gap 达标 → 替换 best 为该嵌入决策。
 *  R8 兜底：异常安全返回原 best，不改坏候选表。
 */
export function applyChampionRule(acts, best, boost, ctx) {
    try {
        if (!acts || !acts.length || !(boost > 0)) return { best: best, replaced: false, hit: 0 };  // R1
        if (!STORE.embeds) return { best: best, replaced: false, hit: 0 };                         // R1
        const gapLimit = (ctx && typeof ctx.gapLimit === 'number') ? ctx.gapLimit : 14;            // R4 阈值

        let hitCount = 0;
        for (let i = 0; i < acts.length; i++) {
            const a = acts[i];
            if (!a) continue;
            const row = getEmbedding(a.type, a.id);   // R2 同类型 + R3 id 精确命中
            if (row && row.value > 0) {               // R5 价值为正
                const bridge = Number(boost) || 0;
                a.score = Math.round((a.score || 0) + bridge * (0.5 + Math.max(0, Math.min(1, row.value))));
                a._champion = true;
                hitCount++;
            } else if (a._champion) {
                delete a._champion;
            }
        }
        if (hitCount === 0) return { best: best, replaced: false, hit: 0 };  // R3 无命中

        const sorted = acts.slice().sort(function (x, y) {
            return (y.score || 0) - (x.score || 0);
        });
        if (!sorted.length) return { best: best, replaced: false, hit: hitCount };  // R4
        const gap = (typeof sorted[0].score === 'number' && typeof sorted[1] && typeof sorted[1].score === 'number')
            ? (sorted[0].score - sorted[1].score) : 1e9;

        let newBest = sorted[0];
        if (newBest && typeof newBest._champion === 'boolean' && newBest._champion) {
            if (gap <= gapLimit) {
                return { best: newBest, replaced: true, hit: hitCount };  // R7 替换成功
            }
        }
        return { best: (newBest === best) ? best : (best || newBest), replaced: (newBest === best), hit: hitCount };
    } catch (e) { return { best: best, replaced: false, hit: 0 }; }  // R8
}

/* ★ 兼容旧接口：仅加权（按 id 精确命中），供其它调用方使用 */
export function applyChampionBoost(acts, boost) {
    try {
        if (!acts || !acts.length || !(boost > 0)) return 0;
        if (!STORE.embeds) return 0;
        let hit = 0;
        for (let i = 0; i < acts.length; i++) {
            const a = acts[i];
            if (!a) continue;
            const row = getEmbedding(a.type, a.id);
            if (row && row.value > 0) {
                a.score = Math.round((a.score || 0) + boost);
                a._champion = true;
                hit++;
            }
        }
        return hit;
    } catch (e) { return 0; }
}

/* ★ 统计信息 */
export function stats() {
    const eb = STORE.embeds || {};
    const keys = Object.keys(eb);
    let total = 0, embCount = 0;
    for (let i = 0; i < keys.length; i++) {
        const rows = eb[keys[i]] || [];
        for (let j = 0; j < rows.length; j++) {
            total += (rows[j].count || 0);
            if (rows[j].emb) embCount++;
        }
    }
    return { types: keys.length, decisions: total, embs: embCount, updatedAt: STORE.updatedAt || 0, embeds: eb };
}

export function clear() {
    try {
        STORE = { embeds: {}, updatedAt: 0 };
        localStorage.removeItem(STORE_KEY);
    } catch (e) {}
}