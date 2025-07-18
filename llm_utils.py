import openai

# Replace this with your actual OpenAI API key
openai.api_key = "your_openai_api_key"

def get_llm_explanation(data, confidence, batting_team, bowling_team, target, venue):
    """
    Generate a strategic explanation for the predicted match outcome.
    """

    response = f"""
**MATCH PREDICTION SUMMARY**
--------------------------------
**Batting Team:** {batting_team}\n
**Bowling Team:** {bowling_team}\n
**Venue:** {venue}\n
**Target:** {target}\n
**Model Confidence Score:** {confidence:.2f}  

**MATCH STATS**
----------------------------- 
**Overs Completed:** {data['over_ball'][0]:.1f}\n
**Runs Scored:** {data['current_score'][0]}\n
**Wickets Lost:** {data['wickets'][0]}\n
**Runs Left:** {data['runs_left'][0]}\n
**Balls Left:** {data['balls_left'][0]}\n 
**Current Run Rate (CRR):** {data['crr'][0]:.2f}\n 
**Required Run Rate (RRR):** {data['rrr'][0]:.2f}\n 

**LLM Strategic Insight:**

The model takes into account the current scenario including overs completed, runs left, wickets lost, and compares CRR with RRR to estimate the winning probability. If the CRR is much lower than RRR and many wickets are lost, the chances reduce significantly. However, strong batting line-ups or small grounds like Wankhede can increase the chance of successful chases even with high targets.

Remember, cricket is unpredictable, and a few explosive overs or quick wickets can shift the momentum entirely!
    """

    return response
