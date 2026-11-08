/* Temple Run 2 HUD — score / coins / distance / multiplier / powerup timers.
 *
 * Draws from the rewrite Env using the shipped UI textures exported to
 * resources/templerun2/ui/ (see resources/templerun2/ASSET_CONTRACT.md).  The
 * sprite catalogue is the APK `interface_new` / HUD atlas: coin, coin_value,
 * gem, score_multiplier, power_meter, powerup_{magnet,shield,boost,bolt,
 * gem_bonus,coin_bonus,score_bonus}, boost_distance, pause, and the stone
 * counter bar (HUDSpiritCove / HUDSpooky / HUDPirateCove / HUDArctic).  The
 * phone gameplay frame recon/visual/frames/phone/phone_01_r1_t82.5s.png pins
 * the score bar top-right over the coin bar, with the coin icon to the RIGHT
 * of the coin number and the pause plaque in the bottom-right corner; this
 * file follows that.  Every asset is optional: when a texture is absent an
 * equivalent procedural glyph and raylib text are drawn instead, so the HUD is
 * always readable.
 *
 * Header-only, C99, all symbols static.  The renderer calls only this file's
 * tr2_hud_init()/tr2_hud_draw(); it must not draw a second HUD of its own.
 */
#ifndef TR2_HUD_H
#define TR2_HUD_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "raylib.h"

/* Directory holding the exported UI sprites/font (see the asset contract). */
#ifndef TR2_HUD_DIR
#define TR2_HUD_DIR "resources/templerun2/ui"
#endif

/* ------------------------------------------------------------------ *
 * Shipped sprite slots.  Each slot is resolved to a file by its exact
 * lowercase stem ui/<stem>.png (plus the shipped underscore/case aliases),
 * most specific slot first (e.g. powerup_coin_bonus before coin), so an
 * unrelated sprite such as bk-coins_1 is never mistaken for an icon.
 * ------------------------------------------------------------------ */
enum {
    TR2_HUD_ICO_COIN_BONUS = 0,
    TR2_HUD_ICO_GEM_BONUS,
    TR2_HUD_ICO_SCORE_BONUS,
    TR2_HUD_ICO_MULT,
    TR2_HUD_ICO_COIN_VALUE,
    TR2_HUD_ICO_COIN,
    TR2_HUD_ICO_GEM,
    TR2_HUD_ICO_METER,
    TR2_HUD_ICO_MAGNET,
    TR2_HUD_ICO_SHIELD,
    TR2_HUD_ICO_BOLT,
    TR2_HUD_ICO_BOOST_DIST,
    TR2_HUD_ICO_BOOST,
    TR2_HUD_ICO_PAUSE,
    TR2_HUD_ICO_PANEL,
    TR2_HUD_ICO_COUNT
};

/* Exact file stems per slot (lowercase, without extension), in priority
 * order.  The prop exporter writes each shipped sprite to ui/<sprite>.png, so
 * an exact ui/<stem>.png match is unambiguous; aliases cover the shipped
 * capitalisation / underscore variants. */
static const char* const tr2_hud_ico_keys[TR2_HUD_ICO_COUNT][6] = {
    {"powerup_coin_bonus", "coinbonus",  "coin_bonus",  NULL, NULL, NULL},
    {"powerup_gem_bonus",  "gembonus",   "gem_bonus",   NULL, NULL, NULL},
    {"powerup_score_bonus","scorebonus", "score_bonus", NULL, NULL, NULL},
    {"score_multiplier",   "scoremultiplier", "multiplier", NULL, NULL, NULL},
    {"coin_value",         "coinvalue",  NULL,          NULL, NULL, NULL},
    {"coin",               NULL,         NULL,          NULL, NULL, NULL},
    {"gem",                NULL,         NULL,          NULL, NULL, NULL},
    {"power_meter",        "powermeter", "multimeter",  "recharge_meter",
     "powerbar",           "meter"},
    {"powerup_magnet",     "magnet",     NULL,          NULL, NULL, NULL},
    {"powerup_shield",     "shield",     NULL,          NULL, NULL, NULL},
    {"powerup_bolt",       "bolt",       NULL,          NULL, NULL, NULL},
    {"boost_distance",     "boostdistance", NULL,       NULL, NULL, NULL},
    {"powerup_boost",      "boost",      NULL,          NULL, NULL, NULL},
    {"pause",              "icon_pause", NULL,          NULL, NULL, NULL},
    /* The stone bar behind the score/coin counters.  The gameplay frame
     * recon/visual/frames/phone/phone_01_r1_t82.5s.png shows the shipped bar
     * with its pink/magenta bevel and green-gem end caps, which is
     * HUDSpiritCove.png; the other HUD* files are the per-region reskins. */
    {"hudspiritcove",      "hudpiratecove", "hudspooky", "hudarctic", "hud",
     NULL},
};

static Texture2D tr2_hud_tex[TR2_HUD_ICO_COUNT];
static Font tr2_hud_font;
static int tr2_hud_font_ok = 0;
static int tr2_hud_loaded = 0;

static void tr2_hud_lower(char* dst, size_t n, const char* src) {
    size_t i = 0;
    for (; src[i] && i + 1 < n; i++) dst[i] = (char)tolower((unsigned char)src[i]);
    dst[i] = '\0';
}

static int tr2_hud_contains(const char* hay, const char* needle) {
    return needle != NULL && strstr(hay, needle) != NULL;
}

/* True when `full` is exactly `core` + ".png". */
static int tr2_hud_is_png_core(const char* full, const char* core) {
    size_t lc = strlen(core);
    return strncmp(full, core, lc) == 0 && strcmp(full + lc, ".png") == 0;
}

static void tr2_hud_init(void) {
    if (tr2_hud_loaded) return;
    tr2_hud_loaded = 1;
    memset(tr2_hud_tex, 0, sizeof(tr2_hud_tex));

    const char* dir = TR2_HUD_DIR;
    if (!DirectoryExists(dir)) return;

    FilePathList files = LoadDirectoryFiles(dir);
    if (files.count == 0) {
        UnloadDirectoryFiles(files);
        return;   /* nothing to load; keep text fallback */
    }

    int n = (int)files.count;
    char (*stem)[96] = (char (*)[96])calloc((size_t)n, sizeof(*stem));
    unsigned char* is_png = (unsigned char*)calloc((size_t)n, sizeof(*is_png));
    unsigned char* claimed = (unsigned char*)calloc((size_t)n, sizeof(*claimed));
    if (stem == NULL || is_png == NULL || claimed == NULL) {
        free(stem); free(is_png); free(claimed);
        UnloadDirectoryFiles(files);
        return;
    }
    for (int i = 0; i < n; i++) {
        const char* path = files.paths[i];
        const char* base = path ? strrchr(path, '/') : NULL;
        base = base ? base + 1 : path;
        if (base == NULL) { stem[i][0] = '\0'; is_png[i] = 0; continue; }
        /* Full name incl. extension, lowercased.  Raw textures (tex_*) and
         * font atlases (font_*) are excluded from sprite matching. */
        tr2_hud_lower(stem[i], sizeof(stem[i]), base);
        is_png[i] = (unsigned char)(tr2_hud_contains(stem[i], ".png")
            && strncmp(stem[i], "tex_", 4) != 0
            && strncmp(stem[i], "font_", 5) != 0);
    }

    /* Claim files for sprite slots, most specific slot first, by exact
     * ui/<stem>.png name.  Substring matching is deliberately avoided: the
     * export tree holds hundreds of unrelated sprites (bk-coins_1, ...) that
     * a loose match would mis-take for an icon. */
    for (int slot = 0; slot < TR2_HUD_ICO_COUNT; slot++) {
        for (int k = 0; k < 6 && tr2_hud_ico_keys[slot][k] != NULL; k++) {
            const char* key = tr2_hud_ico_keys[slot][k];
            int hit = -1;
            for (int i = 0; i < n; i++) {
                if (claimed[i] || !is_png[i]) continue;
                if (tr2_hud_is_png_core(stem[i], key)) { hit = i; break; }
            }
            if (hit < 0) continue;
            claimed[hit] = 1;
            Texture2D t = LoadTexture(files.paths[hit]);
            if (IsTextureValid(t)) {
                tr2_hud_tex[slot] = t;
                break;
            }
        }
    }

    /* Shipped HUD font.  The HUD bundle hud_spiritscove_piratecove_music
     * .unity3d ships Poppins-BlackItalic, and the score/coin numerals in the
     * phone frame are bold and slanted, so prefer it — then the other Poppins
     * weights, then any shipped font.  The Angular score face is a bitmap/SDF
     * atlas with no .fnt, so it cannot be loaded here. */
    {
        static const char* const font_keys[] = {
            "font_poppins-blackitalic.ttf", "font_poppins-black.ttf",
            "font_poppins-extrabold.ttf",   "font_poppins-bold.ttf",
            "font_poppins-semibold.ttf",    NULL
        };
        for (int k = 0; font_keys[k] != NULL && !tr2_hud_font_ok; k++) {
            for (int i = 0; i < n && !tr2_hud_font_ok; i++) {
                if (strcmp(stem[i], font_keys[k]) != 0) continue;
                Font f = LoadFont(files.paths[i]);
                if (IsFontValid(f)) { tr2_hud_font = f; tr2_hud_font_ok = 1; }
            }
        }
        for (int i = 0; i < n && !tr2_hud_font_ok; i++) {
            if (!tr2_hud_contains(stem[i], "poppins")
                    || !tr2_hud_contains(stem[i], ".ttf")) continue;
            Font f = LoadFont(files.paths[i]);
            if (IsFontValid(f)) { tr2_hud_font = f; tr2_hud_font_ok = 1; }
        }
        for (int i = 0; i < n && !tr2_hud_font_ok; i++) {
            if (!tr2_hud_contains(stem[i], ".ttf")
                    && !tr2_hud_contains(stem[i], ".fnt")) continue;
            Font f = LoadFont(files.paths[i]);
            if (IsFontValid(f)) { tr2_hud_font = f; tr2_hud_font_ok = 1; }
        }
    }
    free(stem);
    free(is_png);
    free(claimed);

    UnloadDirectoryFiles(files);
}

static float tr2_hud_u(void) {
    float h = (float)GetScreenHeight();
    return h > 1.0f ? h / 720.0f : 1.0f;
}

static float tr2_hud_text(const char* s, float x, float y, float size, Color c) {
    if (tr2_hud_font_ok) {
        DrawTextEx(tr2_hud_font, s, (Vector2){x, y}, size, size * 0.04f, c);
        return MeasureTextEx(tr2_hud_font, s, size, size * 0.04f).x;
    }
    DrawText(s, (int)x, (int)y, (int)size, c);
    return (float)MeasureText(s, (int)size);
}

static float tr2_hud_measure(const char* s, float size) {
    if (tr2_hud_font_ok) return MeasureTextEx(tr2_hud_font, s, size, size * 0.04f).x;
    return (float)MeasureText(s, (int)size);
}

static void tr2_hud_center(const char* s, float cx, float y, float size, Color c) {
    tr2_hud_text(s, cx - tr2_hud_measure(s, size) * 0.5f, y, size, c);
}

/* Draw a shipped sprite scaled to `h` pixels tall.  Returns 1 when the real
 * texture was drawn, 0 when the caller should draw a procedural glyph. */
static int tr2_hud_icon(int slot, float x, float y, float h, Color tint) {
    if (slot < 0 || slot >= TR2_HUD_ICO_COUNT) return 0;
    Texture2D t = tr2_hud_tex[slot];
    if (!IsTextureValid(t)) return 0;
    float w = h * (float)t.width / (float)t.height;
    DrawTexturePro(t,
        (Rectangle){0.0f, 0.0f, (float)t.width, (float)t.height},
        (Rectangle){x, y, w, h}, (Vector2){0.0f, 0.0f}, 0.0f, tint);
    return 1;
}

/* Natural width of the shipped stone HUD bar at height `h`; falls back to the
 * bar aspect the phone frame shows (~3.6:1) when the sprite is absent. */
static float tr2_hud_bar_w(float h) {
    Texture2D t = tr2_hud_tex[TR2_HUD_ICO_PANEL];
    if (IsTextureValid(t)) return h * (float)t.width / (float)t.height;
    return h * 3.6f;
}

/* Draw the shipped stone/pink counter bar behind the score/coin readouts.
 * Returns 1 when the real sprite was drawn, 0 when a plain backing was used. */
static int tr2_hud_bar(float x, float y, float w, float h) {
    Texture2D t = tr2_hud_tex[TR2_HUD_ICO_PANEL];
    if (IsTextureValid(t)) {
        DrawTexturePro(t,
            (Rectangle){0.0f, 0.0f, (float)t.width, (float)t.height},
            (Rectangle){x, y, w, h}, (Vector2){0.0f, 0.0f}, 0.0f, WHITE);
        return 1;
    }
    DrawRectangleRounded((Rectangle){x, y, w, h}, 0.5f, 8,
        (Color){38, 28, 24, 170});
    return 0;
}

/* Procedural stand-in: a filled disc with a letter, so an absent texture is
 * still legible. */
static void tr2_hud_glyph(float x, float y, float h, Color fill,
        const char* letter) {
    float cx = x + h * 0.5f, cy = y + h * 0.5f;
    DrawCircle((int)cx, (int)cy, h * 0.5f, fill);
    DrawCircleLines((int)cx, (int)cy, h * 0.5f, (Color){20, 16, 12, 200});
    tr2_hud_center(letter, cx, cy - h * 0.30f, h * 0.62f, WHITE);
}

static void tr2_hud_icon_or_glyph(int slot, float x, float y, float h,
        Color fill, const char* letter) {
    if (!tr2_hud_icon(slot, x, y, h, WHITE))
        tr2_hud_glyph(x, y, h, fill, letter);
}

static const char* tr2_hud_death_name(int cause) {
    switch (cause) {
        case TR2_DEATH_FALL:        return "FELL";
        case TR2_DEATH_EATEN:       return "EATEN";
        case TR2_DEATH_FIRE:        return "BURNED";
        case TR2_DEATH_MINEDUCK:    return "MINE - DUCK";
        case TR2_DEATH_MINELEDGE:   return "MINE - LEDGE";
        case TR2_DEATH_MINE:        return "MINE";
        case TR2_DEATH_WHEEL:       return "WHEEL";
        case TR2_DEATH_WATERLEFT:   return "WATER - LEFT";
        case TR2_DEATH_LEDGE:       return "FELL OFF LEDGE";
        case TR2_DEATH_SCENERYTREE: return "TREE";
        case TR2_DEATH_SCENERYROCK: return "ROCK";
        case TR2_DEATH_DEATHTHROW:  return "DEATH THROW";
        case TR2_DEATH_WATERRIGHT:  return "WATER - RIGHT";
        case TR2_DEATH_NONE:        return "RUN ENDED";
        default:                    return "RUN OVER";
    }
}

static const Color TR2_HUD_GOLD   = {255, 226, 140, 255};
static const Color TR2_HUD_COIN   = {246, 196, 54, 255};
static const Color TR2_HUD_ORANGE = {255, 190, 80, 255};
static const Color TR2_HUD_WHITE  = {232, 230, 220, 255};
static const Color TR2_HUD_GEM    = {130, 230, 130, 255};
static const Color TR2_HUD_MAGNET = {230, 80, 80, 255};
static const Color TR2_HUD_SHIELD = {80, 165, 245, 255};
static const Color TR2_HUD_BOOST  = {255, 220, 90, 255};
static const Color TR2_HUD_INVULN = {255, 140, 140, 255};

static void tr2_hud_powerup_row(float x, float y, int slot,
        Color col, const char* letter, const char* label, float remain,
        float span) {
    float u = tr2_hud_u();
    char buf[48];
    snprintf(buf, sizeof(buf), "%s %d", label, (int)(remain + 0.5f));
    float ih = 30.0f * u;
    tr2_hud_icon_or_glyph(slot, x, y, ih, col, letter);
    tr2_hud_text(buf, x + ih + 8.0f * u, y + 4.0f * u, 22.0f * u, col);
    float frac = (span > 0.0f) ? remain / span : 0.0f;
    if (frac < 0.0f) frac = 0.0f;
    if (frac > 1.0f) frac = 1.0f;
    DrawRectangle((int)x, (int)(y + ih + 2.0f * u), (int)(96.0f * u),
        (int)(6.0f * u), (Color){0, 0, 0, 130});
    DrawRectangle((int)x, (int)(y + ih + 2.0f * u),
        (int)(96.0f * u * frac), (int)(6.0f * u), col);
}

static void tr2_hud_draw(Env* env) {
    if (!tr2_hud_loaded) tr2_hud_init();
    int w = GetScreenWidth();
    int h = GetScreenHeight();
    float u = tr2_hud_u();
    char buf[64];

    /* Top-right counters (score over coins).  The shipped HUD pins a stone
     * counter bar to the right edge: the running score on top, the coin count
     * on the bar below it with the coin icon to the RIGHT of the number, then
     * the gem readout (see recon/visual/frames/phone/phone_01_r1_t82.5s.png). */
    {
        float right = (float)w - 6.0f * u;
        float score_h = 56.0f * u;
        float score_y = 6.0f * u;
        float score_fs = 44.0f * u;

        /* Score on its bar. */
        snprintf(buf, sizeof(buf), "%d", (int)env->score);
        float stw = tr2_hud_measure(buf, score_fs);
        float score_w = tr2_hud_bar_w(score_h);
        if (score_w < stw + 30.0f * u) score_w = stw + 30.0f * u;
        float score_x = right - score_w;
        tr2_hud_bar(score_x, score_y, score_w, score_h);
        tr2_hud_text(buf, right - 16.0f * u - stw,
            score_y + (score_h - score_fs) * 0.62f, score_fs, TR2_HUD_GOLD);

        /* Coin count on the bar below, number then coin icon to its right. */
        float coin_h = 44.0f * u;
        float coin_y = score_y + score_h + 4.0f * u;
        float coin_w = tr2_hud_bar_w(coin_h);
        {
            float ih = 30.0f * u;
            float ix = right - 14.0f * u - ih;
            float iy = coin_y + (coin_h - ih) * 0.5f;
            snprintf(buf, sizeof(buf), "%d", env->coins);
            float ctw = tr2_hud_measure(buf, 30.0f * u);
            tr2_hud_bar(right - coin_w, coin_y, coin_w, coin_h);
            tr2_hud_text(buf, ix - 8.0f * u - ctw,
                coin_y + (coin_h - 30.0f * u) * 0.62f, 30.0f * u,
                TR2_HUD_COIN);
            tr2_hud_icon_or_glyph(TR2_HUD_ICO_COIN, ix, iy, ih,
                TR2_HUD_COIN, "C");
        }

        /* Gem readout under the coin bar, only once gems are held. */
        if (env->gems > 0) {
            float ih = 24.0f * u;
            float gy = coin_y + coin_h + 4.0f * u;
            snprintf(buf, sizeof(buf), "%d", env->gems);
            float gtw = tr2_hud_measure(buf, 24.0f * u);
            float gx = right - 14.0f * u - gtw;
            tr2_hud_text(buf, gx, gy, 24.0f * u, TR2_HUD_GEM);
            tr2_hud_icon_or_glyph(TR2_HUD_ICO_GEM, gx - ih - 6.0f * u, gy,
                ih, TR2_HUD_GEM, "G");
        }

        /* Multiplier badge, left of the score bar, shown once it climbs. */
        if (env->multiplier > 1) {
            float ih = 30.0f * u;
            float my = score_y + (score_h - ih) * 0.5f;
            float mx = score_x - 8.0f * u - ih;
            snprintf(buf, sizeof(buf), "x%d", env->multiplier);
            if (!tr2_hud_icon(TR2_HUD_ICO_MULT, mx, my, ih, WHITE))
                tr2_hud_glyph(mx, my, ih, TR2_HUD_ORANGE, "x");
            tr2_hud_text(buf, mx + ih + 6.0f * u, my - 2.0f * u, 26.0f * u,
                TR2_HUD_ORANGE);
        }
    }

    /* Pause button (bottom-right corner, as in the shipped HUD) ------ */
    {
        float ph = 34.0f * u;
        float px = (float)w - 20.0f * u - ph;
        float py = (float)h - 20.0f * u - ph;
        if (!tr2_hud_icon(TR2_HUD_ICO_PAUSE, px, py, ph, WHITE)) {
            DrawRectangleRounded((Rectangle){px, py, ph, ph}, 0.25f, 6,
                (Color){0, 0, 0, 110});
            DrawRectangle((int)(px + ph * 0.28f), (int)(py + ph * 0.22f),
                (int)(ph * 0.15f), (int)(ph * 0.56f), TR2_HUD_WHITE);
            DrawRectangle((int)(px + ph * 0.57f), (int)(py + ph * 0.22f),
                (int)(ph * 0.15f), (int)(ph * 0.56f), TR2_HUD_WHITE);
        }
    }

    /* Coins / gems are drawn with the top-right counter stack above. */

    /* Active powerup timers (bottom-left stack) ---------------------- */
    {
        float x = 24.0f * u;
        float row = 44.0f * u;
        int rows = 0;
        if (env->magnet_timer > 0.0f) rows++;
        if (env->shield_timer > 0.0f) rows++;
        if (env->boost_distance_left > 0.0f) rows++;
        if (env->invuln_timer > 0.0f) rows++;
        float y = (float)h - 24.0f * u - row * (float)rows;
        if (env->magnet_timer > 0.0f) {
            tr2_hud_powerup_row(x, y, TR2_HUD_ICO_MAGNET, TR2_HUD_MAGNET,
                "M", "MAGNET", env->magnet_timer, env->magnet_time);
            y += row;
        }
        if (env->shield_timer > 0.0f) {
            tr2_hud_powerup_row(x, y, TR2_HUD_ICO_SHIELD, TR2_HUD_SHIELD,
                "S", "SHIELD", env->shield_timer, env->shield_time);
            y += row;
        }
        if (env->boost_distance_left > 0.0f) {
            tr2_hud_powerup_row(x, y, TR2_HUD_ICO_BOOST_DIST, TR2_HUD_BOOST,
                "B", "BOOST", env->boost_distance_left, env->boost_distance);
            y += row;
        }
        if (env->invuln_timer > 0.0f) {
            tr2_hud_powerup_row(x, y, TR2_HUD_ICO_BOOST, TR2_HUD_INVULN,
                "V", "INVULN", env->invuln_timer, env->invuln_time);
            y += row;
        }
    }

    /* Slide indicator (bottom-centre) -------------------------------- */
    if (env->sliding) {
        float cx = (float)w * 0.5f;
        float y = (float)h - 40.0f * u;
        float tw = tr2_hud_measure("SLIDE", 24.0f * u);
        DrawRectangleRounded((Rectangle){cx - tw * 0.5f - 14.0f * u,
            y - 6.0f * u, tw + 28.0f * u, 36.0f * u}, 0.4f, 8,
            (Color){0, 0, 0, 120});
        tr2_hud_center("SLIDE", cx, y, 24.0f * u, (Color){150, 220, 255, 255});
        float ax = cx - 4.0f * u;
        DrawTriangle((Vector2){ax - 8.0f * u, y - 16.0f * u},
            (Vector2){ax + 8.0f * u, y - 16.0f * u},
            (Vector2){ax, y - 6.0f * u}, (Color){150, 220, 255, 220});
    }

    /* Death overlay -------------------------------------------------- */
    if (env->dead) {
        const char* name = tr2_hud_death_name(env->death_cause);
        tr2_hud_center(name, (float)w * 0.5f, (float)h * 0.5f - 30.0f * u,
            44.0f * u, (Color){240, 90, 70, 255});
        tr2_hud_center("YOU DIED", (float)w * 0.5f, (float)h * 0.5f + 24.0f * u,
            22.0f * u, TR2_HUD_WHITE);
    }
}

#endif /* TR2_HUD_H */
