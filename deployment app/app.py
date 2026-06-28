"""
FIFA World Cup 2026 — Match Outcome Predictor
Streamlit App (Bonus deployment task)
Run: streamlit run deployment/app.py
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestClassifier


st.set_page_config(
    page_title=" FIFA WC 2026 Predictor",
    page_icon="⚽",
    layout="centered",
)

@st.cache_resource
def load_and_train():
    NAME_MAP = {
        'United States':'USA','United States of America':'USA',
        'South Korea':'Korea Republic','Korea DPR':'North Korea',
        'China':'China PR','Ivory Coast':"Côte d'Ivoire","Cote d'Ivoire":"Côte d'Ivoire",
        'IR Iran':'Iran','Islamic Republic of Iran':'Iran','Holland':'Netherlands',
        'Czech Republic':'Czechia','Republic of Ireland':'Ireland',
        'Bosnia & Herzegovina':'Bosnia-Herzegovina','Bosnia and Herzegovina':'Bosnia-Herzegovina',
        'Trinidad & Tobago':'Trinidad and Tobago','Cabo Verde':'Cape Verde',
        'Congo DR':'DR Congo','Türkiye':'Turkey',
        'Serbia and Montenegro':'Serbia','Federal Republic of Yugoslavia':'Yugoslavia',
    }
    def standardise(name):
        if pd.isna(name): return name
        return NAME_MAP.get(str(name).strip(), str(name).strip())

    matches   = pd.read_csv('data/matches_1930_2022.csv')
    ranking22 = pd.read_csv('data/fifa_ranking_2022-10-06.csv')
    ranking26 = pd.read_csv('data/fifa_ranking_2026-06-08.csv')
    wc_meta   = pd.read_csv('data/world_cup.csv')

    for col in ['home_team','away_team']: matches[col] = matches[col].apply(standardise)
    ranking22['team'] = ranking22['team'].apply(standardise)
    ranking26['team'] = ranking26['team'].apply(standardise)

    cols = ['Year','Round','Date','home_team','away_team','home_score','away_score']
    df = matches[cols].copy()
    df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce')
    df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce')
    df.dropna(subset=['home_score','away_score'], inplace=True)
    df['Year'] = df['Year'].astype(int)

    def encode_result(r):
        if r['home_score'] > r['away_score']: return 0
        elif r['home_score'] < r['away_score']: return 1
        else: return 2
    df['Match_Result'] = df.apply(encode_result, axis=1)
    df['Total_Goals']  = df['home_score'] + df['away_score']

    events = []
    for _, row in df.iterrows():
        events += [{'Date':row['Date'],'team':row['home_team'],'scored':row['home_score'],'conceded':row['away_score']},
                   {'Date':row['Date'],'team':row['away_team'],'scored':row['away_score'],'conceded':row['home_score']}]
    ev = pd.DataFrame(events).sort_values(['team','Date'])
    ev['cum_scored']   = ev.groupby('team')['scored'].transform(lambda x: x.shift(1).expanding().mean())
    ev['cum_conceded'] = ev.groupby('team')['conceded'].transform(lambda x: x.shift(1).expanding().mean())
    ev_h = ev.rename(columns={'team':'home_team','cum_scored':'home_cum_scored','cum_conceded':'home_cum_conceded'})\
             .groupby(['Date','home_team'])[['home_cum_scored','home_cum_conceded']].mean().reset_index()
    ev_a = ev.rename(columns={'team':'away_team','cum_scored':'away_cum_scored','cum_conceded':'away_cum_conceded'})\
             .groupby(['Date','away_team'])[['away_cum_scored','away_cum_conceded']].mean().reset_index()
    df = df.merge(ev_h, on=['Date','home_team'], how='left')
    df = df.merge(ev_a, on=['Date','away_team'], how='left')

    def prior_wins(team, year):
        return sum(1 for _, r in wc_meta.iterrows() if standardise(r['Champion'])==team and int(r['Year'])<year)

    df['home_prior_wins'] = df.apply(lambda r: prior_wins(r['home_team'],r['Year']), axis=1)
    df['away_prior_wins'] = df.apply(lambda r: prior_wins(r['away_team'],r['Year']), axis=1)

    df_sorted = df.sort_values('Date').reset_index(drop=True)
    h2h_wins={}; h2h_ratios=[]
    for idx, row in df_sorted.iterrows():
        key=(row['home_team'],row['away_team'])
        hist=h2h_wins.get(key,{'wins':0,'total':0})
        h2h_ratios.append(hist['wins']/hist['total'] if hist['total']>0 else 0.5)
        if key not in h2h_wins: h2h_wins[key]={'wins':0,'total':0}
        h2h_wins[key]['total']+=1
        if row['Match_Result']==0: h2h_wins[key]['wins']+=1
    df_sorted['h2h_home_win_rate']=h2h_ratios
    df=df_sorted.copy()

    rank_map = ranking22.set_index('team')[['rank','points']].to_dict('index')
    def get_rank(t): return rank_map.get(t,{}).get('rank',120)
    def get_pts(t):  return rank_map.get(t,{}).get('points',1000)
    df['home_rank']=df['home_team'].apply(get_rank); df['away_rank']=df['away_team'].apply(get_rank)
    df['home_points']=df['home_team'].apply(get_pts); df['away_points']=df['away_team'].apply(get_pts)
    df['rank_diff']=df['home_rank']-df['away_rank']; df['pts_diff']=df['home_points']-df['away_points']

    FEATURES=['home_rank','away_rank','home_points','away_points','rank_diff','pts_diff',
              'home_cum_scored','home_cum_conceded','away_cum_scored','away_cum_conceded',
              'home_prior_wins','away_prior_wins','h2h_home_win_rate']
    for f in FEATURES: df[f]=df[f].fillna(df[f].mean())

    train = df[df['Year']<=2018]
    X_tr = train[FEATURES]; y_tr = train['Match_Result']
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight='balanced',
                                random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)

    mean_cs = df['home_cum_scored'].mean()
    mean_cc = df['home_cum_conceded'].mean()

    rank26_map = ranking26.set_index('team')[['rank','points']].to_dict('index')

    #26 team 
    teams_2026 = sorted(set(
        ranking26['team'].tolist()
    ))

    return rf, rank26_map, rank_map, mean_cs, mean_cc, teams_2026, wc_meta, standardise

rf, rank26_map, rank_map, mean_cs, mean_cc, teams_2026, wc_meta, standardise = load_and_train()

FEATURES=['home_rank','away_rank','home_points','away_points','rank_diff','pts_diff',
          'home_cum_scored','home_cum_conceded','away_cum_scored','away_cum_conceded',
          'home_prior_wins','away_prior_wins','h2h_home_win_rate']

def prior_wins(team, year):
    return sum(1 for _, r in wc_meta.iterrows() if standardise(r['Champion'])==team and int(r['Year'])<year)

def predict_match(home, away):
    def get_r(t): return rank26_map.get(t,{}).get('rank',120)
    def get_p(t): return rank26_map.get(t,{}).get('points',1000)
    hr,ar=get_r(home),get_r(away); hp,ap=get_p(home),get_p(away)
    row = {
        'home_rank':hr,'away_rank':ar,'home_points':hp,'away_points':ap,
        'rank_diff':hr-ar,'pts_diff':hp-ap,
        'home_cum_scored':mean_cs,'home_cum_conceded':mean_cc,
        'away_cum_scored':mean_cs,'away_cum_conceded':mean_cc,
        'home_prior_wins':prior_wins(home,2026),'away_prior_wins':prior_wins(away,2026),
        'h2h_home_win_rate':0.5
    }
    X = pd.DataFrame([row])[FEATURES]
    probs = rf.predict_proba(X)[0]
    return probs  
#UI
st.title("FIFA World Cup 2026")
st.subheader("Match Outcome Predictor")
st.markdown("Select teams to predict match outcome probabilities.")
st.divider()

col1, col2 = st.columns(2)
with col1:
    st.markdown("### Home Team")
    home_team = st.selectbox("Select Home Team", teams_2026, index=teams_2026.index("Brazil") if "Brazil" in teams_2026 else 0, key="home")

with col2:
    st.markdown("### Away Team")
    away_options = [t for t in teams_2026 if t != home_team]
    away_team = st.selectbox("Select Away Team", away_options,
                              index=away_options.index("Argentina") if "Argentina" in away_options else 0, key="away")

st.divider()
st.markdown(f"### Selected Match: **{home_team} Vs {away_team}**"  )

if st.button("Predict Match Outcome", type="primary", use_container_width=True):
    probs = predict_match(home_team, away_team)
    hw, aw, dr = probs[0]*100, probs[1]*100, probs[2]*100

    st.markdown(f"## {home_team} Vs {away_team}")
    st.markdown("### Predicted Outcome Probabilities")

    c1, c2, c3 = st.columns(3)
    c1.metric(f"{home_team} Win", f"{hw:.1f}%")
    c2.metric("Draw",f"{dr:.1f}%")
    c3.metric(f"{away_team} Win", f"{aw:.1f}%")
#bar GRaph
    fig, ax = plt.subplots(figsize=(6,3))
    labels = [f"{home_team}\nWin", "Draw", f"{away_team}\nWin"]
    values = [hw, dr, aw]
    colors = ['#02d63b',"#4B5156","#F04444"]
    bars = ax.bar(labels, values, color=colors, edgecolor='white', linewidth=1.5)
    ax.bar_label(bars, fmt='%.1f%%', padding=4, fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(values)+15)
    ax.set_ylabel('Probability (%)', fontsize=11)
    ax.set_title('Match Outcome Probability', fontsize=13, fontweight='bold')
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

    def get_r26(t): return rank26_map.get(t,{}).get('rank','N/A')
    def get_p26(t): return rank26_map.get(t,{}).get('points',0)
    st.divider()
    st.markdown("#### FIFA Ranking (June 2026)")
    ri1, ri2 = st.columns(2)
    ri1.info(f"**{home_team}**\n\nRank: #{get_r26(home_team)}\n\nPoints: {get_p26(home_team):.0f}")
    ri2.info(f"**{away_team}**\n\nRank: #{get_r26(away_team)}\n\nPoints: {get_p26(away_team):.0f}")

st.divider()
st.caption("Model: Random Forest Classifier trained on World Cup matches 1930-2018 · Tested on Qatar 2022")
st.divider()
st.caption("Made with ❤️ by Vikas | Data Source: Kaggle FIFA World Cup Dataset")
