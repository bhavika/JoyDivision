select d.Track, d.Mood, s.* from dataset d
inner join subset s
where s.File = d.Track

