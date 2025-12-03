import os
from json import JSONEncoder

import httpagentparser  # for getting the user agent as json
from flask import Flask, render_template, session
from flask import request, redirect, url_for

from myapp.analytics.analytics_data import AnalyticsData, ClickedDoc, get_utc_timestamp
from myapp.search.load_corpus import load_corpus
from myapp.search.objects import Document, StatsDocument, ResultItem
from myapp.search.search_engine import SearchEngine
from myapp.generation.rag import RAGGenerator
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env


# *** for using method to_json in objects ***
def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)
_default.default = JSONEncoder().default
JSONEncoder.default = _default
# end lines ***for using method to_json in objects ***

def log_request_data(analytics_data, query_id=None, found_count=0, selected_model="None", status_code=200):
    """Helping function to log a generic request event and return request_id."""
    user_agent = request.headers.get('User-Agent')
    user_ip = request.remote_addr
    agent = httpagentparser.detect(user_agent)

    session_cookie_name = app.session_cookie_name 
    session_id = request.cookies.get(session_cookie_name, f"Anon_{os.urandom(8).hex()}")

    # Set user session context
    analytics_data.save_session_data(session_id=session_id, user_agent=user_agent, agent=agent, ip_address=user_ip)

    # Set request data
    request_id = analytics_data.get_new_request_id()
    analytics_data.save_request_data(request_id, session_id, query_id, selected_model, found_count, request.url, request.method, status_code)

    return request_id, session_id


# instantiate the Flask application
app = Flask(__name__)

# random 'secret_key' is used for persisting data in secure cookie
app.secret_key = os.getenv("SECRET_KEY")
# open browser dev tool to see the cookies
app.session_cookie_name = os.getenv("SESSION_COOKIE_NAME")

# instantiate our search engine with the corpus:
# load documents corpus into memory.
print("Loading Search Engine for the first time, this might take a while...")
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
file_path = path + "/" + os.getenv("DATA_FILE_PATH")
corpus = load_corpus(file_path)
search_engine = SearchEngine(corpus=corpus)

# instantiate our in memory persistence
analytics_data = AnalyticsData()
# instantiate RAG generator
rag_generator = RAGGenerator()


# Log first element of corpus to verify it loaded correctly:
print("\nCorpus is loaded... \n First element:\n", list(corpus.values())[0])


# Home URL "/"
@app.route('/')
def index():
    print("starting home url /...")

    # flask server creates a session by persisting a cookie in the user's browser.
    # the 'session' object keeps data between multiple requests. Example:
    session['some_var'] = "Some value that is kept in session"

    user_agent = request.headers.get('User-Agent')
    print("Raw user browser:", user_agent)

    request_id, _ = log_request_data(analytics_data)
    session['last_request_id'] = request_id

    user_ip = request.remote_addr
    agent = httpagentparser.detect(user_agent)

    print("Remote IP: {} - JSON user browser {}".format(user_ip, agent))
    print(session)
    return render_template('index.html', page_title="Welcome")

@app.route('/search', methods=['POST'])
def search_form_post():
    search_query = request.form['search-query']
    selected_model = request.form.get('ranking-model', 'TF-IDF')  # default
    
    if not search_query:
        log_request_data(analytics_data, status_code=302)
        return redirect(url_for("index"))

    session['last_search_query'] = search_query
    session['selected_model'] = selected_model

    # We cannot get the results so we will do the following save in /search_results
    # search_id = analytics_data.save_query_terms(search_query)
    # Save the id in session so the GET route can use it
    # session["search_id"] = search_id

    # This POST request is mostly for HTTP session tracking
    log_request_data(analytics_data, status_code=302)

    return redirect(url_for("search_results"))

@app.route('/search_results', methods=['GET'])
def search_results():
    # Retrieve search query and search id from session
    search_query = session.get('last_search_query', '')
    selected_model = session.get('selected_model', 'TF-IDF')

    # search_id = session.get('search_id', None)
    results = search_engine.search(search_query, search_id=None, search_type=selected_model.lower())
    found_count = len(results)

    results_pids = [res.pid for res in results]
    search_id = analytics_data.save_query_terms(search_query, results_pids)
    session['search_id'] = search_id

    print("Search id: ", search_id)
    request_id, _ = log_request_data(analytics_data, query_id=search_id, found_count=len(results), selected_model=selected_model)
    
    session['last_request_id'] = request_id # storing for future clicks

    # generate RAG response based on user query and retrieved results
    rag_response = rag_generator.generate_response(search_query, results)
    print("RAG response:", rag_response)

    found_count = len(results)
    session['last_found_count'] = found_count

    # Dwell time update
    last_clicked_id = session.pop('last_click_id', None)
    if last_clicked_id:
        # Its a return from a document view (end of dwell)
        analytics_data.update_dwell_time(last_clicked_id)

    return render_template(
        'results.html',
        search_query=search_query,
        results_list=results,
        page_title="Results",
        found_counter=found_count,
        rag_response=rag_response,
        selected_model=selected_model
    )


@app.route('/doc_details', methods=['GET'])
def doc_details():
    """
    Show document details page
    ### Replace with your custom logic ###
    """

    # getting request parameters:
    # user = request.args.get('user')
    print("doc details session: ")
    print(session)

    res = session["some_var"]
    print("recovered var from session:", res)

    # get the query string parameters from request
    clicked_doc_id = request.args["pid"]
    print("click in id={}".format(clicked_doc_id))
    search_id = request.args.get("search_id")
    ranking_str = request.args.get("ranking")

    request_id, session_id = log_request_data(analytics_data, query_id=search_id, status_code=200)

    ranking = int(ranking_str) if ranking_str and ranking_str.isdigit() else 0
    click_id = analytics_data.save_click_data(session_id, request_id, clicked_doc_id, search_id, ranking)
    session['last_click_id'] = click_id
    session['doc_detail_search_id'] = search_id

    # store data in statistics table 1
    # if clicked_doc_id in analytics_data.fact_clicks.keys():
    #     analytics_data.fact_clicks[clicked_doc_id] += 1
    # else:
    #     analytics_data.fact_clicks[clicked_doc_id] = 1

    doc = corpus[clicked_doc_id]

    # print("fact_clicks count for id={} is {}".format(clicked_doc_id, analytics_data.fact_clicks[clicked_doc_id]))
    # print(analytics_data.fact_clicks)

    return render_template('doc_details.html', doc=doc, search_id=search_id)


@app.route('/stats', methods=['GET'])
def stats():
    """
    Show simple statistics example. ### Replace with yourdashboard ###
    :return:
    """
    top_docs_data = analytics_data.get_top_n_clicked_documents(n=50)

    docs = []
    for doc_id, count in top_docs_data:
        row: Document = corpus[doc_id]
        if row:
            doc = StatsDocument(pid=row.pid, title=row.title, description=row.description, url=row.url, count=count)
            docs.append(doc)
    
    # simulate sort by ranking
    docs.sort(key=lambda doc: doc.count, reverse=True)
    return render_template('stats.html', clicks_data=docs)


@app.route('/dashboard', methods=['GET'])
def dashboard():

    # Check if ANY data exists
    has_any_data = (
        bool(analytics_data.dim_sessions) or
        bool(analytics_data.dim_queries) or
        bool(analytics_data.fact_requests) or
        bool(analytics_data.fact_clicks)
    )

    if not has_any_data:
        return render_template("dashboard.html", no_data=True)

    # --- KPIs ---
    total_clicks = len(analytics_data.fact_clicks)
    total_searches = len([req for req in analytics_data.fact_requests.values() if req.get('query_id')])

    dwell_times = [c['dwell_time'] for c in analytics_data.fact_clicks.values() if 'dwell_time' in c]
    avg_dwell_time = sum(dwell_times) / len(dwell_times) if dwell_times else 0
    
    kpis = {
        'total_clicks': total_clicks,
        'total_searches': total_searches,
        'avg_dwell_time': f"{avg_dwell_time:.2f} seconds"
    }

    # --- Reports ---
    top_docs_data = analytics_data.get_top_n_clicked_documents()
    top_visited_docs = []

    for doc_id, count in top_docs_data:
        row = corpus.get(doc_id)
        if row:
            top_visited_docs.append(ClickedDoc(doc_id, row.title, count))

    # --- Per-widget data availability checks ---
    has_queries = bool(analytics_data.dim_queries)
    has_clicks = bool(analytics_data.fact_clicks)
    has_requests = bool(analytics_data.fact_requests)
    has_missions = bool(analytics_data.dim_missions)
    has_searches = bool([req for req in analytics_data.fact_requests.values() if req.get('query_id')])

    # Only generate charts if data exists
    term_count_chart_html = (
        analytics_data.plot_term_count_distribution(
            analytics_data.get_query_term_count_distribution()
        )
        if has_queries else None
    )

    doc_views_chart_html = (
        analytics_data.plot_number_of_views(top_visited_docs)
        if has_clicks else None
    )

    queries_per_mission_chart_html = (
        analytics_data.plot_queries_per_mission(analytics_data.dim_missions)
        if analytics_data.dim_missions else None
    )

    missions_per_session_chart_html = (
        analytics_data.plot_missions_per_session(analytics_data.dim_missions)
        if analytics_data.dim_missions else None
    )

    # Example: timeline for the first mission (optional)
    mission_timeline_chart_html = None
    if analytics_data.dim_missions:
        first_mission_id = next(iter(analytics_data.dim_missions.keys()))
        mission_timeline_chart_html = analytics_data.plot_mission_timeline(
            first_mission_id, analytics_data.dim_missions, analytics_data.dim_queries
        )

    return render_template(
        'dashboard.html',
        no_data=False,
        kpis=kpis,
        visited_docs=top_visited_docs,
        term_count_chart=term_count_chart_html,
        doc_views_chart=doc_views_chart_html,
        has_queries=has_queries,
        has_clicks=has_clicks,
        has_requests=has_requests,
        has_searches=has_searches, 
        has_missions=has_missions,
        top_queries_data=analytics_data.get_top_queries(10) if has_queries else None,
        top_terms=analytics_data.get_top_terms(10) if has_queries else None,
        top_ips=analytics_data.get_top_ips() if has_requests else None,
        browser_chart_route=url_for('plot_preferred_browsers_route') if has_requests else None,
        queries_per_mission_chart=queries_per_mission_chart_html,
        missions_per_session_chart=missions_per_session_chart_html,
        mission_timeline_chart=mission_timeline_chart_html,
    )

# New route added for generating an examples of basic Altair plot (used for dashboard)
@app.route('/plot_number_of_views', methods=['GET'])
def plot_number_of_views():
    top_docs_data = analytics_data.get_top_n_clicked_documents()
    top_visited_docs = []

    for doc_id, count in top_docs_data:
        row = corpus.get(doc_id)
        if row:
            top_visited_docs.append(ClickedDoc(doc_id, row.title, count))
    return analytics_data.plot_number_of_views(top_visited_docs)

@app.route('/plot_term_count_distribution', methods=['GET'])
def plot_term_count_distribution_route():
    df = analytics_data.get_query_term_count_distribution()
    return analytics_data.plot_term_count_distribution(df)

@app.route('/plot_preferred_browsers', methods=['GET'])
def plot_preferred_browsers_route():
    df = analytics_data.get_preferred_browsers()
    return analytics_data.plot_preferred_browsers(df)


if __name__ == "__main__":
    app.run(port=8088, host="0.0.0.0", threaded=False, debug=os.getenv("DEBUG"))
