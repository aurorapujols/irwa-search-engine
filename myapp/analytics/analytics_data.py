import json
import random
import altair as alt
import pandas as pd
import uuid

from collections import Counter
from datetime import datetime, timezone, timedelta
from myapp.search.objects import StatsDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class ClickedDoc:
    def __init__(self, doc_id, description, counter):
        self.doc_id = doc_id
        self.description = description
        self.counter = counter

    def to_json(self):
        return self.__dict__

    def __str__(self):
        """
        Print the object content as a JSON string
        """
        return json.dumps(self)


def get_utc_timestamp():
    return datetime.now(timezone.utc)

class AnalyticsData:
    """
    An in memory persistence object.
    Declare more variables to hold analytics tables.
    """
    # Example of statistics table
    # fact_clicks is a dictionary with the click counters: key = doc id | value = click counter
    #fact_clicks = dict([])

    ### Please add your custom tables here:

    # FACT TABLE Click [key: click_id] Value: {'session_id', 'request_id', 'doc_id', 'query_id', 'ranking_at_click', 'click_time', 'dwell_start_time'}
    fact_clicks: dict[str, dict] = {}

    # FACT TABLE Request [key: request_id] Value: {'session_id', 'query_id', 'model_used', 'found_count', 'request_time', 'url', 'method', 'statusa_code'}
    fact_requests: dict[str, dict] = {}

    # FACT TABLE Query [key: mission_id] Value: {'session_id', 'queries_ids'}
    dim_missions: dict[str, dict] = {}
    
    # DIMENSION TABLE Session [key: session_id] Value: {'user_agent', 'browser', 'os', 'device', 'ip_address', 'start_time', 'end_time'}
    dim_sessions: dict[str, dict] = {}

    # DIMENSION TABLE Query [key: query_id] Value: {'query_terms', 'term_count', 'processed_time', 'result_ids'}
    dim_queries: dict[str, dict] = {}

    # DIMENSION TABLE Document [key: doc_id (pid)] Value: {'doc_title'}
    dim_documents: dict[str, dict] = {}


    vectorizer = TfidfVectorizer()

    def get_new_request_id(self):
        """Generate a unique ID for a new request."""
        return str(uuid.uuid4())
    
    def save_session_data(self, session_id, user_agent, agent, ip_address):
        """Saves or updates session data"""
        if session_id not in self.dim_sessions:
            self.dim_sessions[session_id] = {
                'user_agent': user_agent,
                'browser': agent.get('browser', {}).get('name', 'Unknown'),
                'os': agent.get('os', {}).get('name', 'Unknown'),
                'device': 'Mobile' if agent.get('is_mobile') else 'Desktop',
                'ip_address': ip_address,
                'start_time': get_utc_timestamp(),
                'end_time': None
            }
        else:
            self.dim_sessions[session_id]['end_time'] = get_utc_timestamp()
    
    def save_query_terms(self, terms, result_pids):
        """Saves query information and returns a new query ID"""
        query_id = str(uuid.uuid4())
        self.dim_queries[query_id] = {
            'query_terms': terms,
            'term_count': len(terms.split()),
            'processed_time': get_utc_timestamp(),
            'result_ids': result_pids
        }
        return query_id
    
    def save_request_data(self, request_id, session_id, query_id, model_used, found_count, url, method, status_code=200):
        """Saves request information for a search request."""
        self.fact_requests[request_id] = {
            'session_id': session_id,
            'query_id': query_id,
            'model_used': model_used,
            'found_count': found_count,
            'request_time': get_utc_timestamp(),
            'url': url,
            'method': method,
            'status_code': status_code
        }

    def save_click_data(self, session_id, request_id, doc_id, query_id, ranking):
        """Saves click information (start of dwell time)."""
        click_id = str(uuid.uuid4())
        self.fact_clicks[click_id] = {
            'session_id': session_id,
            'request_id': request_id,
            'doc_id': doc_id,
            'query_id': query_id,
            'ranking_at_click': ranking,
            'click_time': get_utc_timestamp(),
            'dwell_start_time': get_utc_timestamp()
        }
        return click_id
    
    def update_dwell_time(self, click_id):
        """Updates the click record to compute dwell time."""
        if click_id in self.fact_clicks:
            start_time = self.fact_clicks[click_id]['dwell_start_time']
            end_time = get_utc_timestamp()

            dwell_time = (end_time - start_time).total_seconds()
            self.fact_clicks[click_id]['dwell_time'] = dwell_time
            print(f"Dwell time for click {click_id}: {dwell_time:.2f} seconds")

    def save_mission(self, session_id, query_id, min_time_window=10, similarity_thres=0.75):
        now = get_utc_timestamp()
        query_terms = self.dim_queries[query_id]["query_terms"]
        
        # Select candidate queries within session and time window
        candidates = []
        for req_id, req_data in self.fact_requests.items():
            if (req_data['session_id']==session_id) and (req_data['query_id']!=query_id):
                time_diff = now - self.dim_queries[query_id]['processed_time']
                if time_diff <= timedelta(minutes=min_time_window):
                    candidates.append([query_id, self.dim_queries[query_id]])

        # Check similarity
        if candidates:
            terms = [q_data['query_terms'] for _,q_data in candidates] + [query_terms]
            tfidf = self.vectorizer.fit_transform(terms)
            similarities = linear_kernel(tfidf[-1], tfidf[:-1])[0]

            for i, [q_id, _] in enumerate(candidates): 
                if similarities[i] >= similarity_thres:
                    for m_id, m_data in self.dim_missions.items():
                        if q_id in m_data['queries_ids']:
                            m_data['queries_ids'].append(query_id)
                            return m_id
        else:
            mission_id = self.get_new_request_id()
            self.dim_missions[mission_id] = {'session_id':session_id, 'queries_ids':[query_id]}
            return mission_id

    # def save_query_terms(self, terms: str) -> int:
    #     print(self)
    #     return random.randint(0, 100000)

    # PLOTS -----------------------------------------------------------------------------------------------------------------#
    
    # def plot_number_of_views(self):
    #     # Prepare data
    #     data = [{'Document ID': doc_id, 'Number of Views': count} for doc_id, count in self.fact_clicks.items()]
    #     df = pd.DataFrame(data)
    #     # Create Altair chart
    #     chart = alt.Chart(df).mark_bar().encode(
    #         x='Document ID',
    #         y='Number of Views'
    #     ).properties(
    #         title='Number of Views per Document'
    #     )
    #     # Render the chart to HTML
    #     return chart.to_html()

    def plot_number_of_views(self, top_visited_docs:dict[ClickedDoc]):
        """Generates the Altair chart for Top 10 clicked documents."""
        
        # Data preparation: Convert list of ClickedDoc objects to a DataFrame
        # Ensure the columns here match the field names in the chart below.
        data = [
            {
                'DocumentID': doc.doc_id, 
                'DocumentTitle': doc.description, # Using 'DocumentTitle' (no space)
                'Views': doc.counter
            } 
            for doc in top_visited_docs
        ]
        df = pd.DataFrame(data)

        # We must explicitly limit the DataFrame to avoid overly long charts
        df = df.sort_values(by='Views', ascending=False).head(10)

        # Altair Chart Creation: Use the EXACT column names defined above
        chart = alt.Chart(df).mark_bar().encode(
            # X-axis: Number of Views, Quantitative data type (Q)
            x=alt.X('Views:Q', title='Total Document Clicks'),
            
            # Y-axis: Document Title, Nominal data type (N), sorted by Views
            y=alt.Y('DocumentTitle:N', sort=alt.SortField(field='Views', order='descending'), title='Document Title'),
            
            # Tooltip for interaction
            tooltip=['DocumentTitle:N', 'Views:Q']
        ).properties(
            title='Top 10 Most Clicked Documents'
        ).interactive()

        return chart.to_html()
    
    def plot_term_count_distribution(self, df: pd.DataFrame):
        """Generates the Altair chart for Query Term Count Distribution."""
        df.columns = ['Term Count', 'Frequency']
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('Term Count', type='ordinal'),
            y='Frequency',
            tooltip=['Term Count', 'Frequency']
        ).properties(
            title='Distribution of Query Term Count'
        ).interactive()
        
        return chart.to_html()


    def get_top_n_clicked_documents(self, n=10):
        click_counts = {}
        for click in self.fact_clicks.values():
            doc_id = click['doc_id']
            click_counts[doc_id] = click_counts.get(doc_id, 0) + 1

        # Sort
        sorted_docs = sorted(click_counts.items(), key=lambda item: item[1], reverse=True)
        return sorted_docs[:n]


    def get_query_term_count_distribution(self):
        term_counts = [q['term_count'] for q in self.dim_queries.values()]
        df = pd.DataFrame(term_counts, columns=['Term Count'])
        return df['Term Count'].value_counts().sort_index().reset_index()
    
  
    def get_preferred_browsers(self, n=5):
        """Calculates and returns the top N preferred browsers."""
        browser_counts = {}
        
        # Aggregate browser data from the session dimension table
        for session_data in self.dim_sessions.values():
            browser = session_data.get('browser', 'Unknown')
            browser_counts[browser] = browser_counts.get(browser, 0) + 1
        
        # Convert to DataFrame for easy Altair plotting
        df = pd.DataFrame(browser_counts.items(), columns=['Browser', 'Frequency'])
        
        # Sort and take top N
        df = df.sort_values(by='Frequency', ascending=False).head(n)
        return df


    def plot_preferred_browsers(self, df: pd.DataFrame):
        """Generates the Altair chart for Preferred Browsers."""
        chart = alt.Chart(df).mark_arc(outerRadius=120).encode(
            theta=alt.Theta("Frequency", stack=True),
            color=alt.Color("Browser"),
            tooltip=['Browser', 'Frequency']
        ).properties(
            title='Visitor Preferred Browsers'
        ).interactive()
        
        return chart.to_html()
    
  
    def get_top_queries(self, n=10):
        """Calculates and returns the top N most frequent search queries."""
        query_counts = {}
        
        # Count occurrences of query_id in Fact_Request table
        for request_data in self.fact_requests.values():
            query_id = request_data.get('query_id')
            if query_id and query_id in self.dim_queries:
                query_text = self.dim_queries[query_id]['query_terms']
                query_counts[query_text] = query_counts.get(query_text, 0) + 1
        
        # Sort and take top N
        sorted_queries = sorted(query_counts.items(), key=lambda item: item[1], reverse=True)
        return sorted_queries[:n]


    def get_top_terms(self, n=10):
        """Calculates and returns the top N most frequent single terms used in searches."""
        term_counts = {}
        
        # Process terms from all Dim_Query entries
        for query_data in self.dim_queries.values():
            terms = query_data['query_terms'].lower().split()
            for term in terms:
                # Basic cleaning: remove punctuation
                clean_term = ''.join(filter(str.isalnum, term))
                if clean_term:
                    term_counts[clean_term] = term_counts.get(clean_term, 0) + 1
        
        # Sort and take top N
        sorted_terms = sorted(term_counts.items(), key=lambda item: item[1], reverse=True)
        return sorted_terms[:n]


    def get_top_ips(self, n=5):
        """Calculates and returns the top N most frequent IP addresses."""
        # Ensure ip_counts is initialized inside this local scope
        ip_counts = {} 
        
        # Check for sessions and aggregate
        if self.dim_sessions:
            for session_data in self.dim_sessions.values():
                # We assume the IP address is stored in the session data
                ip = session_data.get('ip_address', 'Unknown')
                ip_counts[ip] = ip_counts.get(ip, 0) + 1
        
        # Sort the results. ip_counts is guaranteed to be a dictionary (empty or full).
        sorted_ips = sorted(ip_counts.items(), key=lambda item: item[1], reverse=True)
        return sorted_ips[:n]


    def plot_queries_per_mission(missions):
        data = [
            {"mission_id": mid, "query_count": len(m["query_ids"])}
            for mid, m in missions.items()
        ]
        df = pd.DataFrame(data)

        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("mission_id:N", title="Mission ID"),
                y=alt.Y("query_count:Q", title="Number of Queries")
            )
            .properties(title="Queries per Mission", width=400, height=300)
        )

        return chart.to_html()
    
    def plot_missions_per_session(missions):
        counts = Counter(m["session_id"] for m in missions.values())

        df = pd.DataFrame({
            "session_id": list(counts.keys()),
            "mission_count": list(counts.values())
        })

        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("session_id:N", title="Session ID"),
                y=alt.Y("mission_count:Q", title="Number of Missions")
            )
            .properties(title="Missions per Session", width=400, height=300)
        )

        return chart.to_html()
       
    def plot_mission_timeline(mission_id, missions, dim_queries):
        mission = missions[mission_id]
        query_ids = mission["query_ids"]

        df = pd.DataFrame({
            "query_id": query_ids,
            "time": [dim_queries[q]["processed_time"] for q in query_ids]
        }).sort_values("time")

        df["order"] = range(1, len(df) + 1)

        chart = (
            alt.Chart(df)
            .mark_line(point=True)
            .encode(
                x=alt.X("time:T", title="Timestamp"),
                y=alt.Y("order:Q", title="Query Order"),
                tooltip=["query_id", "time"]
            )
            .properties(title=f"Timeline of Mission {mission_id}", width=500, height=300)
        )

        return chart.to_html()
    # PLOTS -----------------------------------------------------------------------------------------------------------------#

