from prevision import api_predicthq

a = api_predicthq()
attendanceDf = a.get_today_df_attendance()
print(attendanceDf.head())
