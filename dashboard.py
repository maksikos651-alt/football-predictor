import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="AI Football Sniper", layout="wide", page_icon="⚽")


# --- 1. FUNKCJE SILNIKA (Backend) ---

@st.cache_data(ttl=3600)
def get_upcoming_fixtures(league_name):
    url = "https://www.football-data.co.uk/fixtures.csv"
    try:
        df = pd.read_csv(url, encoding='utf-8-sig', on_bad_lines='skip')
        df.columns = df.columns.str.strip()

        if 'Div' not in df.columns:
            for col in df.columns:
                if 'Div' in col or 'Division' in col:
                    df = df.rename(columns={col: 'Div'})
                    break
        if 'Div' not in df.columns: return pd.DataFrame()

        league_map = {
            "Premier League": "E0", "Championship": "E1", "League One": "E2", "League Two": "E3",
            "La Liga": "SP1", "La Liga 2": "SP2", "Bundesliga": "D1", "Bundesliga 2": "D2",
            "Serie A": "I1", "Serie B": "I2", "Ligue 1": "F1", "Ligue 2": "F2",
            "Eredivisie": "N1", "Jupiler League": "B1", "Liga Portugal": "P1",
            "Super Lig": "T1", "Greece Super League": "G1", "Scottish Premiership": "SC0"
        }
        div_code = league_map.get(league_name)
        league_fixtures = df[df['Div'] == div_code].copy()
        league_fixtures['Date'] = pd.to_datetime(league_fixtures['Date'], dayfirst=True, errors='coerce')

        today = pd.Timestamp.now().normalize()
        future_games = league_fixtures[league_fixtures['Date'] >= today]
        return future_games.sort_values(['Date', 'Time'])
    except:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_and_prep_data(league_name):
    # BEZPIECZNE POŁĄCZENIE
    # Kod szuka zmiennej "DB_URL" w sekretach.
    # Lokalnie: bierze z pliku .streamlit/secrets.toml
    # W Chmurze: bierze z ustawień Advanced Settings

    try:
        db_url = st.secrets["DB_URL"]
    except FileNotFoundError:
        st.error("❌ Błąd konfiguracji: Nie znaleziono sekretów bazy danych!")
        st.stop()

    engine = create_engine(db_url)

    query = f"SELECT * FROM matches WHERE league = '{league_name}' ORDER BY date ASC"
    df = pd.read_sql(query, engine)

    df = df.rename(columns={
        'date': 'Date', 'home_team': 'HomeTeam', 'away_team': 'AwayTeam',
        'home_goals': 'FTHG', 'away_goals': 'FTAG', 'result': 'FTR',
        'odds_home': 'B365H', 'odds_draw': 'B365D', 'odds_away': 'B365A',
        'odds_over25': 'B365_O25', 'odds_under25': 'B365_U25',
        'home_corners': 'HC', 'away_corners': 'AC'
    })
    return df


def add_rolling_features(df, window=5):
    data = df.copy()
    home_df = data[['Date', 'HomeTeam', 'FTHG', 'FTAG', 'FTR', 'HC', 'AC']].rename(
        columns={'HomeTeam': 'Team', 'FTHG': 'GS', 'FTAG': 'GC', 'HC': 'CorW', 'AC': 'CorL'})
    home_df['Pts'] = np.where(home_df['FTR'] == 'H', 3, np.where(home_df['FTR'] == 'D', 1, 0))
    home_df['IsHome'] = 1

    away_df = data[['Date', 'AwayTeam', 'FTAG', 'FTHG', 'FTR', 'AC', 'HC']].rename(
        columns={'AwayTeam': 'Team', 'FTAG': 'GS', 'FTHG': 'GC', 'AC': 'CorW', 'HC': 'CorL'})
    away_df['Pts'] = np.where(away_df['FTR'] == 'A', 3, np.where(away_df['FTR'] == 'D', 1, 0))
    away_df['IsHome'] = 0

    team_stats = pd.concat([home_df, away_df]).sort_values('Date')
    features = team_stats.groupby('Team')[['GS', 'GC', 'Pts', 'CorW']].transform(
        lambda x: x.rolling(window, min_periods=3).mean().shift(1)
    )
    team_stats[['Form_Att', 'Form_Def', 'Form_Pts', 'Form_Cor']] = features

    h_stats = team_stats[team_stats['IsHome'] == 1][
        ['Date', 'Team', 'Form_Att', 'Form_Def', 'Form_Pts', 'Form_Cor']].rename(
        columns={'Team': 'HomeTeam', 'Form_Att': 'Home_Att', 'Form_Def': 'Home_Def', 'Form_Pts': 'Home_Form',
                 'Form_Cor': 'Home_Corners_Avg'})
    a_stats = team_stats[team_stats['IsHome'] == 0][
        ['Date', 'Team', 'Form_Att', 'Form_Def', 'Form_Pts', 'Form_Cor']].rename(
        columns={'Team': 'AwayTeam', 'Form_Att': 'Away_Att', 'Form_Def': 'Away_Def', 'Form_Pts': 'Away_Form',
                 'Form_Cor': 'Away_Corners_Avg'})

    data = pd.merge(data, h_stats, on=['Date', 'HomeTeam'], how='left')
    data = pd.merge(data, a_stats, on=['Date', 'AwayTeam'], how='left')
    data['OddsDiff'] = (1 / data['B365H']) - (1 / data['B365A'])
    return data.dropna()


@st.cache_resource
def train_model(df, prediction_type="1X2"):
    predictors = ['OddsDiff', 'Home_Att', 'Away_Att', 'Home_Def', 'Away_Def', 'Home_Form', 'Away_Form',
                  'Home_Corners_Avg', 'Away_Corners_Avg']
    best_params = {'n_estimators': 300, 'learning_rate': 0.01, 'max_depth': 4, 'subsample': 0.7,
                   'colsample_bytree': 0.8, 'n_jobs': -1}

    if prediction_type == "1X2":
        predictors += ['B365H', 'B365A', 'B365D']
        df['Target'] = df['FTR'].map({'A': 0, 'D': 1, 'H': 2})
        model = xgb.XGBClassifier(objective='multi:softprob', num_class=3, **best_params)
        model.fit(df[predictors], df['Target'])
    else:
        predictors += ['B365_O25', 'B365_U25']
        target = np.where((df['FTHG'] + df['FTAG']) > 2.5, 1, 0)
        model = xgb.XGBClassifier(objective='binary:logistic', **best_params)
        model.fit(df[predictors], target)
    return model, predictors


def calculate_kelly(prob, odds, bankroll, fraction):
    if prob * odds <= 1: return 0.0
    b = odds - 1
    return bankroll * max(((b * prob - (1 - prob)) / b), 0) * fraction


def plot_score_heatmap(home_exp, away_exp):
    probs = np.zeros((5, 5))
    for h in range(5):
        for a in range(5):
            probs[a, h] = poisson.pmf(h, home_exp) * poisson.pmf(a, away_exp) * 100
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(probs, annot=True, fmt=".1f", cmap="YlGnBu", cbar=False, ax=ax)
    ax.set_xlabel("Gole Gospodarza")
    ax.set_ylabel("Gole Gościa")
    ax.set_title("Dokładny Wynik (%)")
    return fig


# --- 2. INTERFEJS GŁÓWNY ---

st.title("⚽ AI Football Sniper")

# --- PANEL BOCZNY (KONFIGURACJA) ---
st.sidebar.header("Ustawienia Systemu")

league_config = {
    "Bundesliga": {"roi_1x2": 0.29, "roi_ou": -14.85, "recom": "🟡 MAŁY ZYSK (Ostrożnie)"},
    "Bundesliga 2": {"roi_1x2": -9.44, "roi_ou": -25.80, "recom": "⛔ UNIKAJ TEJ LIGI"},
    "Championship": {"roi_1x2": 14.31, "roi_ou": -3.81, "recom": "🔥 GRAJ ZWYCIĘZCĘ (1X2)"},
    "Eredivisie": {"roi_1x2": 26.59, "roi_ou": -18.47, "recom": "🔥 GRAJ ZWYCIĘZCĘ (1X2)"},
    "Greece Super League": {"roi_1x2": 38.11, "roi_ou": 20.05, "recom": "🔥 GRAJ WSZYSTKO"},
    "Jupiler League": {"roi_1x2": -29.22, "roi_ou": 9.95, "recom": "🔥 GRAJ GOLE (O/U)"},
    "La Liga": {"roi_1x2": 5.95, "roi_ou": 44.56, "recom": "🔥 GRAJ GOLE (O/U)"},
    "La Liga 2": {"roi_1x2": -14.14, "roi_ou": 9.80, "recom": "🔥 GRAJ GOLE (O/U)"},
    "League One": {"roi_1x2": 6.36, "roi_ou": -14.27, "recom": "🔥 GRAJ ZWYCIĘZCĘ (1X2)"},
    "League Two": {"roi_1x2": -15.54, "roi_ou": 11.74, "recom": "🔥 GRAJ GOLE (O/U)"},
    "Liga Portugal": {"roi_1x2": -18.10, "roi_ou": -0.18, "recom": "⛔ UNIKAJ TEJ LIGI"},
    "Ligue 1": {"roi_1x2": -43.09, "roi_ou": -9.61, "recom": "⛔ UNIKAJ TEJ LIGI"},
    "Ligue 2": {"roi_1x2": -36.91, "roi_ou": -28.99, "recom": "⛔ UNIKAJ TEJ LIGI"},
    "Premier League": {"roi_1x2": -9.73, "roi_ou": -22.39, "recom": "⛔ UNIKAJ TEJ LIGI"},
    "Scottish Premiership": {"roi_1x2": 34.72, "roi_ou": -9.43, "recom": "🔥 GRAJ ZWYCIĘZCĘ (1X2)"},
    "Serie A": {"roi_1x2": 49.42, "roi_ou": -15.18, "recom": "🔥 GRAJ ZWYCIĘZCĘ (1X2)"},
    "Serie B": {"roi_1x2": -20.08, "roi_ou": 13.05, "recom": "🔥 GRAJ GOLE (O/U)"},
    "Super Lig": {"roi_1x2": -1.37, "roi_ou": 17.20, "recom": "🔥 GRAJ GOLE (O/U)"},
}

selected_league = st.sidebar.selectbox("Wybierz Ligę", sorted(list(league_config.keys())))
bet_type = st.sidebar.radio("Strategia", ["Zwycięzca (1X2)", "Gole (Over/Under 2.5)"])
bankroll = st.sidebar.number_input("Bankroll (PLN)", 1000)
kelly_frac = st.sidebar.slider("Kelly %", 0.05, 0.2, 0.1)

# Status ligi
cfg = league_config.get(selected_league, {"roi_1x2": 0, "roi_ou": 0, "recom": "?"})
st.sidebar.info(f"{cfg['recom']}")
c1, c2 = st.sidebar.columns(2)
c1.metric("ROI 1X2", f"{cfg['roi_1x2']}%")
c2.metric("ROI O/U", f"{cfg['roi_ou']}%")

# ŁADOWANIE DANYCH
with st.spinner("Ładowanie AI..."):
    raw_data = load_and_prep_data(selected_league)
    if raw_data.empty: st.error("Brak danych"); st.stop()
    processed_data = add_rolling_features(raw_data)
    train_mode = "1X2" if bet_type == "Zwycięzca (1X2)" else "OU25"
    model, features = train_model(processed_data, train_mode)

# --- ZAKŁADKI ---
tab1, tab2, tab3 = st.tabs(["🔥 Skaner Value", "🔍 Analiza Meczu", "📜 Historia i Weryfikacja"])

# --- TAB 1: SKANER VALUE (AUTOMATYCZNY) ---
with tab1:
    st.header(f"Skaner Okazji: {selected_league}")

    fixtures = get_upcoming_fixtures(selected_league)

    if fixtures.empty:
        st.info("Brak nadchodzących meczów w pliku fixtures.csv. Sprawdź w piątek!")
    else:
        value_bets = []
        progress = st.progress(0)
        total_games = len(fixtures)

        # ... (to jest wewnątrz with tab1:) ...

        for i, (index, row) in enumerate(fixtures.iterrows()):

            # Pasek postępu
            current_prog = (i + 1) / total_games
            progress.progress(min(current_prog, 1.0))

            # Znajdź statystyki
            h_stats = processed_data[processed_data['HomeTeam'] == row['HomeTeam']]
            a_stats = processed_data[processed_data['AwayTeam'] == row['AwayTeam']]

            if h_stats.empty or a_stats.empty: continue

            h_stat = h_stats.iloc[-1]
            a_stat = a_stats.iloc[-1]

            # Budujemy bazowy DataFrame
            match_input = pd.DataFrame([{
                'Home_Att': h_stat['Home_Att'], 'Away_Att': a_stat['Away_Att'],
                'Home_Def': h_stat['Home_Def'], 'Away_Def': a_stat['Away_Def'],
                'Home_Form': h_stat['Home_Form'], 'Away_Form': a_stat['Away_Form'],
                'Home_Corners_Avg': h_stat['Home_Corners_Avg'], 'Away_Corners_Avg': a_stat['Away_Corners_Avg']
            }])

            if bet_type == "Zwycięzca (1X2)":
                o1 = row.get('B365H');
                ox = row.get('B365D');
                o2 = row.get('B365A')
                # Zabezpieczenie przed NaN (puste kursy)
                if pd.isna(o1) or pd.isna(ox) or pd.isna(o2): continue

                match_input['OddsDiff'] = (1 / o1) - (1 / o2)
                match_input['B365H'] = o1;
                match_input['B365D'] = ox;
                match_input['B365A'] = o2

                # --- KLUCZOWA POPRAWKA ---
                # Sortujemy kolumny tak, jak chce tego model!
                match_input = match_input[features]
                # -------------------------

                probs = model.predict_proba(match_input)[0]

                for outcome, prob, odd, name in [(2, probs[2], o1, row['HomeTeam']),
                                                 (0, probs[0], o2, row['AwayTeam'])]:
                    val = (prob * odd) - 1
                    if val > 0.05:
                        value_bets.append({
                            "Data": row['Date'].strftime('%d.%m'),
                            "Mecz": f"{row['HomeTeam']} vs {row['AwayTeam']}",
                            "Typ": name,
                            "Kurs": odd,
                            "Szansa AI": f"{prob * 100:.1f}%",
                            "Value": f"{val * 100:.1f}%",
                            "Rekomendacja": "🔥 GRAJ" if val > 0.1 else "✅ Warto"
                        })
            else:
                # O/U
                oo = row.get('B365>2.5');
                ou = row.get('B365<2.5')
                if pd.isna(oo) or pd.isna(ou): continue

                match_input['OddsDiff'] = (1 / oo) - (1 / ou)
                match_input['B365_O25'] = oo;
                match_input['B365_U25'] = ou

                # --- KLUCZOWA POPRAWKA ---
                match_input = match_input[features]
                # -------------------------

                p_over = model.predict_proba(match_input)[0][1]
                p_under = 1.0 - p_over

                val_o = (p_over * oo) - 1
                if val_o > 0.05:
                    value_bets.append(
                        {"Data": row['Date'].strftime('%d.%m'), "Mecz": f"{row['HomeTeam']} vs {row['AwayTeam']}",
                         "Typ": "OVER 2.5", "Kurs": oo, "Szansa AI": f"{p_over * 100:.1f}%",
                         "Value": f"{val_o * 100:.1f}%", "Rekomendacja": "🔥 GRAJ" if val_o > 0.1 else "✅ Warto"})

                val_u = (p_under * ou) - 1
                if val_u > 0.05:
                    value_bets.append(
                        {"Data": row['Date'].strftime('%d.%m'), "Mecz": f"{row['HomeTeam']} vs {row['AwayTeam']}",
                         "Typ": "UNDER 2.5", "Kurs": ou, "Szansa AI": f"{p_under * 100:.1f}%",
                         "Value": f"{val_u * 100:.1f}%", "Rekomendacja": "🔥 GRAJ" if val_u > 0.1 else "✅ Warto"})

        progress.empty()

        if value_bets:
            st.success(f"Znaleziono {len(value_bets)} okazji w lidze {selected_league}!")
            df_bets = pd.DataFrame(value_bets)
            st.dataframe(df_bets, use_container_width=True)
        else:
            st.warning("Brak wyraźnego Value w nadchodzących meczach tej ligi.")

# --- TAB 2: ANALIZA SZCZEGÓŁOWA (STARA WERSJA) ---
with tab2:
    st.header("Analiza Manualna")

    # Wybór meczu
    match_opts = ["Wybierz ręcznie..."]
    match_map = {}
    if not fixtures.empty:
        for idx, row in fixtures.iterrows():
            lbl = f"{row['Date'].strftime('%d.%m')} | {row['HomeTeam']} vs {row['AwayTeam']}"
            match_opts.append(lbl)
            match_map[lbl] = row

    sel_fix = st.selectbox("Wybierz mecz z terminarza", match_opts)

    # Domyślne wartości
    def_h, def_a = 0, 1
    d_o1, d_ox, d_o2 = 2.0, 3.2, 3.5
    d_oo, d_ou = 1.9, 1.9

    if sel_fix != "Wybierz ręcznie...":
        md = match_map[sel_fix]
        teams = sorted(raw_data['HomeTeam'].unique())
        try:
            def_h = teams.index(md['HomeTeam'])
            def_a = teams.index(md['AwayTeam'])
            if pd.notnull(md.get('B365H')): d_o1 = float(md['B365H'])
            if pd.notnull(md.get('B365D')): d_ox = float(md['B365D'])
            if pd.notnull(md.get('B365A')): d_o2 = float(md['B365A'])
            if pd.notnull(md.get('B365>2.5')): d_oo = float(md['B365>2.5'])
            if pd.notnull(md.get('B365<2.5')): d_ou = float(md['B365<2.5'])
        except:
            pass

    col_t1, col_t2 = st.columns(2)
    teams = sorted(raw_data['HomeTeam'].unique())
    with col_t1:
        home_team = st.selectbox("Gospodarz", teams, index=def_h)
    with col_t2:
        away_team = st.selectbox("Gość", teams, index=def_a)

    col_k1, col_k2, col_k3 = st.columns(3)
    if bet_type == "Zwycięzca (1X2)":
        with col_k1:
            o1 = st.number_input("1 (Home)", value=d_o1, step=0.05)
        with col_k2:
            ox = st.number_input("X (Draw)", value=d_ox, step=0.05)
        with col_k3:
            o2 = st.number_input("2 (Away)", value=d_o2, step=0.05)
    else:
        with col_k1:
            oo = st.number_input("Over 2.5", value=d_oo, step=0.05)
        with col_k2:
            ou = st.number_input("Under 2.5", value=d_ou, step=0.05)
        o1, o2, ox = 0, 0, 0  # Dummy

    if st.button("PRZEANALIZUJ TEN MECZ", type="primary"):
        h_stats = processed_data[processed_data['HomeTeam'] == home_team].iloc[-1]
        a_stats = processed_data[processed_data['AwayTeam'] == away_team].iloc[-1]

        # Tworzymy Input
        input_data = pd.DataFrame([{
            'Home_Att': h_stats['Home_Att'], 'Away_Att': a_stats['Away_Att'],
            'Home_Def': h_stats['Home_Def'], 'Away_Def': a_stats['Away_Def'],
            'Home_Form': h_stats['Home_Form'], 'Away_Form': a_stats['Away_Form'],
            'Home_Corners_Avg': h_stats['Home_Corners_Avg'], 'Away_Corners_Avg': a_stats['Away_Corners_Avg']
        }])

        if bet_type == "Zwycięzca (1X2)":
            input_data['OddsDiff'] = (1 / o1) - (1 / o2)
            input_data['B365H'] = o1;
            input_data['B365D'] = ox;
            input_data['B365A'] = o2

            # --- NAPRAWA BŁĘDU (Sortowanie kolumn) ---
            input_data = input_data[features]
            # -----------------------------------------

            probs = model.predict_proba(input_data)[0]
            outcomes = [("GOSPODARZ", probs[2], o1), ("REMIS", probs[1], ox), ("GOŚĆ", probs[0], o2)]
        else:
            input_data['OddsDiff'] = (1 / oo) - (1 / ou)
            input_data['B365_O25'] = oo;
            input_data['B365_U25'] = ou

            # --- NAPRAWA BŁĘDU (Sortowanie kolumn) ---
            input_data = input_data[features]
            # -----------------------------------------

            p_over = model.predict_proba(input_data)[0][1]
            outcomes = [("OVER 2.5", p_over, oo), ("UNDER 2.5", 1 - p_over, ou)]

        # Wyświetlanie wyników
        c_res = st.columns(len(outcomes))
        for i, (lbl, p, o) in enumerate(outcomes):
            with c_res[i]:
                val = (p * o) - 1
                st.metric(lbl, f"{p * 100:.1f}%", f"Value: {val * 100:.1f}%",
                          delta_color="normal" if val > 0 else "off")
                if val > 0.05:
                    stake = calculate_kelly(p, o, bankroll, kelly_frac)
                    st.success(f"Graj za: {stake:.2f} zł")

        st.divider()
        xg_h = (h_stats['Home_Att'] + a_stats['Away_Def']) / 2
        xg_a = (a_stats['Away_Att'] + h_stats['Home_Def']) / 2
        st.write("Heatmapa Wyników:")
        st.pyplot(plot_score_heatmap(xg_h, xg_a))

# --- TAB 3: HISTORIA I WERYFIKACJA ---
with tab3:
    st.header(f"Ostatnie mecze: {selected_league}")
    st.markdown("Sprawdź, jak model poradził sobie w ostatnich 20 meczach.")

    # Bierzemy ostatnie 20 meczów z bazy (te które mają wynik FTR)
    history_data = processed_data.sort_values("Date", ascending=False).head(20).copy()

    history_rows = []

    for idx, row in history_data.iterrows():
        # Rekonstrukcja predykcji
        input_row = pd.DataFrame([row[features]])  # Features są już w df
        input_row['OddsDiff'] = (1 / row['B365H']) - (1 / row['B365A'])  # Refresh

        # Odtwarzamy kursy (muszą być w input)
        if bet_type == "Zwycięzca (1X2)":
            input_row['B365H'] = row['B365H']
            input_row['B365D'] = row['B365D']
            input_row['B365A'] = row['B365A']

            probs = model.predict_proba(input_row)[0]
            # Sprawdzamy co by obstawił (największe Value)
            outcomes = [(2, probs[2], row['B365H'], 'H'), (0, probs[0], row['B365A'], 'A')]
            best_val = 0
            pick = "Brak"
            result_color = "⚪"

            for target, p, o, lbl in outcomes:
                val = (p * o) - 1
                if val > 0.1:  # Value > 10%
                    best_val = val
                    pick = f"{lbl} (@{o})"
                    # Czy weszło?
                    if row['FTR'] == lbl:
                        result_color = "🟢 WIN"
                    else:
                        result_color = "🔴 LOSS"
        else:
            # O/U
            input_row['B365_O25'] = row['B365_O25']
            input_row['B365_U25'] = row['B365_U25']
            p_over = model.predict_proba(input_row)[0][1]

            total_goals = row['FTHG'] + row['FTAG']
            real_res = "O" if total_goals > 2.5 else "U"

            val_o = (p_over * row['B365_O25']) - 1
            val_u = ((1 - p_over) * row['B365_U25']) - 1

            pick = "Brak"
            result_color = "⚪"

            if val_o > 0.1:
                pick = f"OVER (@{row['B365_O25']})"
                result_color = "🟢 WIN" if real_res == "O" else "🔴 LOSS"
            elif val_u > 0.1:
                pick = f"UNDER (@{row['B365_U25']})"
                result_color = "🟢 WIN" if real_res == "U" else "🔴 LOSS"

        history_rows.append({
            "Data": row['Date'].strftime('%d.%m'),
            "Mecz": f"{row['HomeTeam']} vs {row['AwayTeam']}",
            "Wynik": f"{int(row['FTHG'])}-{int(row['FTAG'])}",
            "Model by zagrał": pick,
            "Rozliczenie": result_color
        })

    st.dataframe(pd.DataFrame(history_rows))