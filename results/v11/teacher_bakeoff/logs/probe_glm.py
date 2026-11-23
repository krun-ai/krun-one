import json,urllib.request,sys
P='Task: pick the cheapest option that has at least 8 GB of memory.\n\nCandidates:\n1. a: 4 GB, $1\n2. b: 8 GB, $3\n3. c: 16 GB, $2\n0. none\n\nAnswer with JSON only: {"choice": <number>}'
S={'type':'object','properties':{'choice':{'type':'integer'}},'required':['choice']}
model=sys.argv[1]
for name,extra in [('schema think:false',{'think':False,'format':S}),('format json think:false',{'think':False,'format':'json'}),('schema think:true',{'think':True,'format':S}),('no format think:false',{'think':False})]:
    b={'model':model,'stream':False,'messages':[{'role':'user','content':P}],'options':{'temperature':0,'num_predict':600}}; b.update(extra)
    try:
        d=json.load(urllib.request.urlopen(urllib.request.Request('http://localhost:11434/api/chat',json.dumps(b).encode(),{'Content-Type':'application/json'}),timeout=120))
        m=d.get('message',{}); print(name,'| content=',repr(m.get('content','')[:160]),'| thinking_chars=',len(m.get('thinking') or ''),'| eval',d.get('eval_count'),d.get('done_reason'))
    except Exception as e: print(name,'ERR',e)
