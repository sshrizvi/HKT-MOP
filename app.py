from matplotlib.dates import TU
import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path


# Helper function to load metrics
def load_metrics(filename):
    """Load metrics from JSON file"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"File not found: {filename}")
        return None

# Load all metrics
hkt_standard = load_metrics('experiments/results/hierarchical/hkt_standard_test_metrics.json')
hkt_domain = load_metrics('experiments/results/hierarchical/hkt_domain_shift_test_metrics.json')
dkt_binary_standard = load_metrics('experiments/results//dkt/dkt_binary_standard_test_metrics.json')
dkt_binary_domain = load_metrics('experiments/results//dkt/dkt_binary_domain_shift_test_metrics.json')
dkt_multiclass_standard = load_metrics('experiments/results//dkt/dkt_multiclass_standard_test_metrics.json')
dkt_multiclass_domain = load_metrics('experiments/results//dkt/dkt_multiclass_domain_shift_test_metrics.json')


# Page configuration
st.set_page_config(
    page_title="HKT-MOP Dashboard",
    page_icon="HKT",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for navigation
st.sidebar.title("Dashboard")
view_option = st.sidebar.radio(
    label="Select View",
    options=["Overview", "HKT Model", "DKT Binary Model", "DKT Multiclass Model", "Model Comparison"]
)


# Overview Section
if view_option == "Overview":
    # Title
    st.header(":yellow[_Hierarchical_] Knowledge Tracing on Multi Outcome Programming Data", width=600, anchor=False)
    
    # Models Implementation Details
    with st.container():
        st.markdown("""
            <div style='padding:20px; border-radius:10px; background-color:#ffa520; box-shadow:0 4px 8px rgba(0,0,0,0.1);'>
                <h3>Models Implemented</h3>
                <div style="font-family: Arial, sans-serif; line-height: 1.5;">
                    <p>
                        <strong>1. DKT Binary:</strong>
                        Predicts whether a student will solve a programming exercise correctly or incorrectly using a recurrent neural network.
                    </p>
                    <p>
                        <strong>2. DKT Multiclass:</strong>
                        Predicts the specific outcome (e.g., AC, WA, TLE) of a student's programming submission using a recurrent neural network.
                    </p>
                    <p>
                        <strong>3. HKT Model:</strong>
                        Uses a hierarchical approach to first predict compilation success and then execution outcome for programming submissions, reflecting the natural evaluation pipeline.
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Evaluation Details
    with st.container():
        st.markdown("""
            <div style='margin-top:20px; padding:20px; border-radius:10px; background-color:#813cf5; box-shadow:0 4px 8px rgba(0,0,0,0.1);'>
                <h3>Evaluation Methods</h3>
                <div style="font-family: Arial, sans-serif; max-width: 1000px; line-height: 1.5; display: flex; gap: 20px;">
                    <!-- DKT Binary Evaluation Column -->
                    <div style="flex: 1; padding: 12px; border-left: 4px solid #2e86c1; background: #e6f0fa; color: #1a1a1a;">
                    <h4 style="margin-top: 0; margin-bottom: 8px;">DKT Binary</h4>
                    <ul style="margin: 0 0 0 18px; padding: 0;">
                        <li><strong>Task:</strong> Predicts whether the next interaction is correct or incorrect.</li>
                        <li><strong>Metrics:</strong> Test loss, accuracy, macro F1, and AUC for binary classification.</li>
                        <li><strong>Splits:</strong> Evaluated separately on standard and domain-shift test sets.</li>
                    </ul>
                    </div>
                    <!-- DKT Multiclass Evaluation Column -->
                    <div style="flex: 1; padding: 12px; border-left: 4px solid #28b463; background: #f8fcf7;">
                        <h4 style="margin-top: 0; margin-bottom: 8px; color: #196619;">DKT Multiclass</h4>
                        <ul style="margin: 0 0 0 18px; padding: 0; color: #333;">
                        <li><strong>Task:</strong> Predicts one of multiple outcome classes (e.g., AC, WA, TLE) for the next interaction.</li>
                        <li><strong>Metrics:</strong> Test loss, accuracy, and macro F1 over all outcome classes.</li>
                        <li><strong>Splits:</strong> Uses the same standard and domain-shift evaluation protocol as binary DKT.</li>
                        </ul>
                    </div>
                    <!-- HKT Model Evaluation Column -->
                    <div style="flex: 1; padding: 12px; border-left: 4px solid #ca6f1e; background: #fff8f3;">
                        <h4 style="margin-top: 0; margin-bottom: 8px; color: #945d15;">HKT Model</h4>
                        <ul style="margin: 0 0 0 18px; padding: 0; color: #5a4a27;">
                        <li><strong>Compilation head:</strong> AUC, accuracy, and macro F1 for predicting CE vs compiled.</li>
                        <li><strong>Execution head:</strong> Accuracy and macro F1 over execution outcomes, computed only on compiled submissions.</li>
                        <li><strong>Splits & logs:</strong> Metrics stored as JSON for standard and domain-shift test sets.</li>
                        </ul>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.header("Key Metrics Summary")
    
    # Create summary table
    summary_data = []
    
    if hkt_standard:
        summary_data.append({
            "Model": "HKT",
            "Scenario": "Standard Test",
            "Compilation Acc": f"{hkt_standard['compilation_accuracy']:.4f}",
            "Compilation AUC": f"{hkt_standard['compilation_auc']:.4f}",
            "Execution Acc": f"{hkt_standard['execution_accuracy']:.4f}"
        })
    
    if hkt_domain:
        summary_data.append({
            "Model": "HKT",
            "Scenario": "Domain Shift",
            "Compilation Acc": f"{hkt_domain['compilation_accuracy']:.4f}",
            "Compilation AUC": f"{hkt_domain['compilation_auc']:.4f}",
            "Execution Acc": f"{hkt_domain['execution_accuracy']:.4f}"
        })
    
    if dkt_binary_standard:
        summary_data.append({
            "Model": "DKT Binary",
            "Scenario": "Standard Test",
            "Accuracy": f"{dkt_binary_standard['test_accuracy']:.4f}",
            "AUC": f"{dkt_binary_standard['test_auc']:.4f}",
            "F1 Score": f"{dkt_binary_standard['test_f1']:.4f}"
        })
    
    if dkt_binary_domain:
        summary_data.append({
            "Model": "DKT Binary",
            "Scenario": "Domain Shift",
            "Accuracy": f"{dkt_binary_domain['test_accuracy']:.4f}",
            "AUC": f"{dkt_binary_domain['test_auc']:.4f}",
            "F1 Score": f"{dkt_binary_domain['test_f1']:.4f}"
        })
    
    if dkt_multiclass_standard:
        summary_data.append({
            "Model": "DKT Multiclass",
            "Scenario": "Standard Test",
            "Accuracy": f"{dkt_multiclass_standard['test_accuracy']:.4f}",
            "F1 Score": f"{dkt_multiclass_standard['test_f1']:.4f}",
            "Loss": f"{dkt_multiclass_standard['test_loss']:.4f}"
        })
    
    if dkt_multiclass_domain:
        summary_data.append({
            "Model": "DKT Multiclass",
            "Scenario": "Domain Shift",
            "Accuracy": f"{dkt_multiclass_domain['test_accuracy']:.4f}",
            "F1 Score": f"{dkt_multiclass_domain['test_f1']:.4f}",
            "Loss": f"{dkt_multiclass_domain['test_loss']:.4f}"
        })
    
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True, hide_index=True)


# HKT Model Section
elif view_option == "HKT Model":
    # Title
    st.header(":yellow[_HKT_] Model Performance", width=600, anchor=False)
    
    tab1, tab2 = st.tabs(["Standard", "Domain Shift"])
    
    with tab1:
        if hkt_standard:
            with st.container():
                st.markdown("### Standard Test Metrics")
                
                # Metrics display
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Compilation AUC", f"{hkt_standard['compilation_auc']:.4f}", border=True)
                with col2:
                    st.metric("Compilation Acc", f"{hkt_standard['compilation_accuracy']:.4f}", border=True)
                with col3:
                    st.metric("Compilation F1", f"{hkt_standard['compilation_f1']:.4f}", border=True)
                with col4:
                    st.metric("Execution Acc", f"{hkt_standard['execution_accuracy']:.4f}", border=True)
                with col5:
                    st.metric("Execution F1", f"{hkt_standard['execution_f1']:.4f}", border=True)
            
            st.markdown("### Metric Comparison")
            with st.container(border=True):
                
                # Visualization
                col_a, col_b = st.columns(2)
                
                with col_a:
                    # Compilation metrics
                    fig1 = go.Figure(data=[
                        go.Bar(name='Compilation Metrics', 
                            x=['AUC', 'Accuracy', 'F1 Score'],
                            y=[hkt_standard['compilation_auc'], 
                                hkt_standard['compilation_accuracy'], 
                                hkt_standard['compilation_f1']],
                            marker_color='#813cf5')
                    ])
                    fig1.update_layout(
                        title="Compilation Metrics",
                        yaxis_title="Score",
                        yaxis_range=[0, 1],
                        height=400
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col_b:
                    # Execution metrics
                    fig2 = go.Figure(data=[
                        go.Bar(name='Execution Metrics', 
                            x=['Accuracy', 'F1 Score'],
                            y=[hkt_standard['execution_accuracy'], 
                                hkt_standard['execution_f1']],
                            marker_color='#ffa520')
                    ])
                    fig2.update_layout(
                        title="Execution Metrics",
                        yaxis_title="Score",
                        yaxis_range=[0, 1],
                        height=400
                    )
                    st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        if hkt_domain:
            st.markdown("### Domain Shift Test Metrics")
            
            with st.container():
                # Metrics display
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    delta_auc = hkt_domain['compilation_auc'] - hkt_standard['compilation_auc'] if hkt_standard else None
                    st.metric("Compilation AUC", f"{hkt_domain['compilation_auc']:.4f}", 
                            delta=f"{delta_auc:.4f}" if delta_auc else None, border=True)
                with col2:
                    delta_acc = hkt_domain['compilation_accuracy'] - hkt_standard['compilation_accuracy'] if hkt_standard else None
                    st.metric("Compilation Acc", f"{hkt_domain['compilation_accuracy']:.4f}",
                            delta=f"{delta_acc:.4f}" if delta_acc else None, border=True)
                with col3:
                    delta_f1 = hkt_domain['compilation_f1'] - hkt_standard['compilation_f1'] if hkt_standard else None
                    st.metric("Compilation F1", f"{hkt_domain['compilation_f1']:.4f}",
                            delta=f"{delta_f1:.4f}" if delta_f1 else None, border=True)
                with col4:
                    delta_ex_acc = hkt_domain['execution_accuracy'] - hkt_standard['execution_accuracy'] if hkt_standard else None
                    st.metric("Execution Acc", f"{hkt_domain['execution_accuracy']:.4f}",
                            delta=f"{delta_ex_acc:.4f}" if delta_ex_acc else None, border=True)
                with col5:
                    delta_ex_f1 = hkt_domain['execution_f1'] - hkt_standard['execution_f1'] if hkt_standard else None
                    st.metric("Execution F1", f"{hkt_domain['execution_f1']:.4f}",
                            delta=f"{delta_ex_f1:.4f}" if delta_ex_f1 else None, border=True)
            
            # Comparison visualization
            if hkt_standard:
                st.markdown("### Standard vs Domain Shift Comparison")
                
                with st.container(border=True):
                    metrics = ['Compilation\nAUC', 'Compilation\nAcc', 'Compilation\nF1', 'Execution\nAcc', 'Execution\nF1']
                    standard_vals = [
                        hkt_standard['compilation_auc'],
                        hkt_standard['compilation_accuracy'],
                        hkt_standard['compilation_f1'],
                        hkt_standard['execution_accuracy'],
                        hkt_standard['execution_f1']
                    ]
                    domain_vals = [
                        hkt_domain['compilation_auc'],
                        hkt_domain['compilation_accuracy'],
                        hkt_domain['compilation_f1'],
                        hkt_domain['execution_accuracy'],
                        hkt_domain['execution_f1']
                    ]
                    
                    fig = go.Figure(data=[
                        go.Bar(name='Standard Test', x=metrics, y=standard_vals, marker_color='#ffa520'),
                        go.Bar(name='Domain Shift', x=metrics, y=domain_vals, marker_color="#fe4a4b")
                    ])
                    fig.update_layout(
                        barmode='group',
                        yaxis_title="Score",
                        yaxis_range=[0, 1],
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)

# DKT Binary Model Section
elif view_option == "DKT Binary Model":
    # Title
    st.header(":yellow[_DKT Binary_] Model Performance", width=600, anchor=False)
    
    tab1, tab2 = st.tabs(["Standard", "Domain Shift"])
    
    with tab1:
        if dkt_binary_standard:
            st.markdown("### Standard Test Metrics")
            
            # Metrics display
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Test Loss", f"{dkt_binary_standard['test_loss']:.4f}", border=True)
            with col2:
                st.metric("Accuracy", f"{dkt_binary_standard['test_accuracy']:.4f}", border=True)
            with col3:
                st.metric("AUC", f"{dkt_binary_standard['test_auc']:.4f}", border=True)
            with col4:
                st.metric("F1 Score", f"{dkt_binary_standard['test_f1']:.4f}", border=True)
            
            # Visualization
            st.markdown("### Performance Metrics")
            
            with st.container(border=True):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    # Bar chart
                    fig1 = go.Figure(data=[
                        go.Bar(x=['Accuracy', 'AUC', 'F1 Score'],
                            y=[dkt_binary_standard['test_accuracy'],
                                dkt_binary_standard['test_auc'],
                                dkt_binary_standard['test_f1']],
                            marker_color=['#ffa520', '#813cf5', '#00c1f3'])
                    ])
                    fig1.update_layout(
                        title="Classification Metrics",
                        yaxis_title="Score",
                        yaxis_range=[0, 1],
                        height=400
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col_b:
                    # Radar chart
                    fig2 = go.Figure(data=go.Scatterpolar(
                        r=[dkt_binary_standard['test_accuracy'],
                        dkt_binary_standard['test_auc'],
                        dkt_binary_standard['test_f1']],
                        theta=['Accuracy', 'AUC', 'F1 Score'],
                        fill='toself',
                        name='DKT Binary'
                    ))
                    fig2.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        title="Performance Profile",
                        height=400
                    )
                    st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        if dkt_binary_domain:
            st.markdown("### Domain Shift Test Metrics")
            
            # Metrics display with delta
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                delta_loss = dkt_binary_domain['test_loss'] - dkt_binary_standard['test_loss'] if dkt_binary_standard else None
                st.metric("Test Loss", f"{dkt_binary_domain['test_loss']:.4f}",
                         delta=f"{delta_loss:.4f}" if delta_loss else None,
                         delta_color="inverse", border=True)
            with col2:
                delta_acc = dkt_binary_domain['test_accuracy'] - dkt_binary_standard['test_accuracy'] if dkt_binary_standard else None
                st.metric("Accuracy", f"{dkt_binary_domain['test_accuracy']:.4f}",
                         delta=f"{delta_acc:.4f}" if delta_acc else None, border=True)
            with col3:
                delta_auc = dkt_binary_domain['test_auc'] - dkt_binary_standard['test_auc'] if dkt_binary_standard else None
                st.metric("AUC", f"{dkt_binary_domain['test_auc']:.4f}",
                         delta=f"{delta_auc:.4f}" if delta_auc else None, border=True)
            with col4:
                delta_f1 = dkt_binary_domain['test_f1'] - dkt_binary_standard['test_f1'] if dkt_binary_standard else None
                st.metric("F1 Score", f"{dkt_binary_domain['test_f1']:.4f}",
                         delta=f"{delta_f1:.4f}" if delta_f1 else None, border=True)
            
            # Comparison visualization
            if dkt_binary_standard:
                st.markdown("### Standard vs Domain Shift Comparison")
                
                with st.container(border=True):
                    metrics = ['Accuracy', 'AUC', 'F1 Score']
                    standard_vals = [
                        dkt_binary_standard['test_accuracy'],
                        dkt_binary_standard['test_auc'],
                        dkt_binary_standard['test_f1']
                    ]
                    domain_vals = [
                        dkt_binary_domain['test_accuracy'],
                        dkt_binary_domain['test_auc'],
                        dkt_binary_domain['test_f1']
                    ]
                    
                    fig = go.Figure(data=[
                        go.Bar(name='Standard Test', x=metrics, y=standard_vals, marker_color='#ffa520'),
                        go.Bar(name='Domain Shift', x=metrics, y=domain_vals, marker_color='#fe4a4b')
                    ])
                    fig.update_layout(
                        barmode='group',
                        yaxis_title="Score",
                        yaxis_range=[0, 1],
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)

# DKT Multiclass Model Section
elif view_option == "DKT Multiclass Model":
    # Title
    st.header(":yellow[_DKT MutliClass_] Model Performance", width=600, anchor=False)
    
    tab1, tab2 = st.tabs(["Standard", "Domain Shift"])
    
    with tab1:
        if dkt_multiclass_standard:
            st.markdown("### Standard Test Metrics")
            
            # Metrics display
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Test Loss", f"{dkt_multiclass_standard['test_loss']:.4f}", border=True)
            with col2:
                st.metric("Accuracy", f"{dkt_multiclass_standard['test_accuracy']:.4f}", border=True)
            with col3:
                st.metric("F1 Score", f"{dkt_multiclass_standard['test_f1']:.4f}", border=True)
            
            # Visualization
            st.markdown("### Performance Metrics")
            
            with st.container(border=True):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    # Bar chart
                    fig1 = go.Figure(data=[
                        go.Bar(x=['Accuracy', 'F1 Score'],
                            y=[dkt_multiclass_standard['test_accuracy'],
                                dkt_multiclass_standard['test_f1']],
                            marker_color=['#813cf5', '#1d83e0'])
                    ])
                    fig1.update_layout(
                        title="Classification Metrics",
                        yaxis_title="Score",
                        yaxis_range=[0, 1],
                        height=400
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col_b:
                    # Gauge chart for accuracy
                    fig2 = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=dkt_multiclass_standard['test_accuracy'] * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Accuracy (%)"},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "#17becf"},
                            'steps': [
                                {'range': [0, 50], 'color': "#fe4a4b"},
                                {'range': [50, 75], 'color': "#ffa520"},
                                {'range': [75, 100], 'color': "#3b8a59"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 70.75
                            }
                        }
                    ))
                    fig2.update_layout(height=400)
                    st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        if dkt_multiclass_domain:
            st.markdown("### Domain Shift Test Metrics")
            
            # Metrics display with delta
            col1, col2, col3 = st.columns(3)
            
            with col1:
                delta_loss = dkt_multiclass_domain['test_loss'] - dkt_multiclass_standard['test_loss'] if dkt_multiclass_standard else None
                st.metric("Test Loss", f"{dkt_multiclass_domain['test_loss']:.4f}",
                         delta=f"{delta_loss:.4f}" if delta_loss else None,
                         delta_color="inverse", border=True)
            with col2:
                delta_acc = dkt_multiclass_domain['test_accuracy'] - dkt_multiclass_standard['test_accuracy'] if dkt_multiclass_standard else None
                st.metric("Accuracy", f"{dkt_multiclass_domain['test_accuracy']:.4f}",
                         delta=f"{delta_acc:.4f}" if delta_acc else None, border=True)
            with col3:
                delta_f1 = dkt_multiclass_domain['test_f1'] - dkt_multiclass_standard['test_f1'] if dkt_multiclass_standard else None
                st.metric("F1 Score", f"{dkt_multiclass_domain['test_f1']:.4f}",
                         delta=f"{delta_f1:.4f}" if delta_f1 else None, border=True)
            
            # Comparison visualization
            if dkt_multiclass_standard:
                st.markdown("### Standard vs Domain Shift Comparison")
                
                with st.container(border=True):
                    col_x, col_y = st.columns(2)
                    
                    with col_x:
                        metrics = ['Accuracy', 'F1 Score']
                        standard_vals = [
                            dkt_multiclass_standard['test_accuracy'],
                            dkt_multiclass_standard['test_f1']
                        ]
                        domain_vals = [
                            dkt_multiclass_domain['test_accuracy'],
                            dkt_multiclass_domain['test_f1']
                        ]
                        
                        fig1 = go.Figure(data=[
                            go.Bar(name='Standard Test', x=metrics, y=standard_vals, marker_color='#ffa520'),
                            go.Bar(name='Domain Shift', x=metrics, y=domain_vals, marker_color='#fe4a4b')
                        ])
                        fig1.update_layout(
                            barmode='group',
                            yaxis_title="Score",
                            yaxis_range=[0, 1],
                            height=400
                        )
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col_y:
                        # Loss comparison
                        fig2 = go.Figure(data=[
                            go.Bar(x=['Standard Test', 'Domain Shift'],
                                y=[dkt_multiclass_standard['test_loss'],
                                    dkt_multiclass_domain['test_loss']],
                                marker_color=['#ffa520', '#fe4a4b'])
                        ])
                        fig2.update_layout(
                            title="Test Loss Comparison",
                            yaxis_title="Loss",
                            height=400
                        )
                        st.plotly_chart(fig2, use_container_width=True)


# Model Comparison Section
elif view_option == "Model Comparison":
    # Title
    st.header(":yellow[_Cross-Model_] Performance Comparison", width=600, anchor=False)
    
    comparison_type = st.radio(
        "Select Comparison Type:",
        ["Standard Test Comparison", "Domain Shift Comparison", "Robustness Analysis"],
        horizontal=True
    )
    
    if comparison_type == "Standard Test Comparison":
        st.markdown("### Standard Test Performance Across Models")
        
        # Create comparison dataframe
        comparison_data = []
        
        if hkt_standard:
            comparison_data.append({
                "Model": "HKT",
                "Type": "Multi-task",
                "Primary Accuracy": hkt_standard['compilation_accuracy'],
                "AUC": hkt_standard['compilation_auc'],
                "F1 Score": hkt_standard['compilation_f1']
            })
        
        if dkt_binary_standard:
            comparison_data.append({
                "Model": "DKT Binary",
                "Type": "Binary",
                "Primary Accuracy": dkt_binary_standard['test_accuracy'],
                "AUC": dkt_binary_standard['test_auc'],
                "F1 Score": dkt_binary_standard['test_f1']
            })
        
        if dkt_multiclass_standard:
            comparison_data.append({
                "Model": "DKT Multiclass",
                "Type": "Multiclass",
                "Primary Accuracy": dkt_multiclass_standard['test_accuracy'],
                "AUC": "-",
                "F1 Score": dkt_multiclass_standard['test_f1']
            })
        
        if comparison_data:
            df_comp = pd.DataFrame(comparison_data)
            st.dataframe(df_comp, use_container_width=True, hide_index=True)
            
            # Visualization
            st.markdown("### Accuracy Comparison")
            
            models = [d["Model"] for d in comparison_data]
            accuracies = [d["Primary Accuracy"] for d in comparison_data]
            
            fig = go.Figure(data=[
                go.Bar(x=models, y=accuracies, 
                       marker_color=['#ffa520', '#813cf5', '#00c1f3'],
                       text=[f"{acc:.4f}" for acc in accuracies],
                       textposition='auto')
            ])
            fig.update_layout(
                title="Model Accuracy Comparison (Standard Test)",
                yaxis_title="Accuracy",
                yaxis_range=[0, 1],
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif comparison_type == "Domain Shift Comparison":
        st.markdown("### Domain Shift Performance Across Models")
        
        # Create comparison dataframe
        comparison_data = []
        
        if hkt_domain:
            comparison_data.append({
                "Model": "HKT",
                "Type": "Multi-task",
                "Primary Accuracy": hkt_domain['compilation_accuracy'],
                "AUC": hkt_domain['compilation_auc'],
                "F1 Score": hkt_domain['compilation_f1']
            })
        
        if dkt_binary_domain:
            comparison_data.append({
                "Model": "DKT Binary",
                "Type": "Binary",
                "Primary Accuracy": dkt_binary_domain['test_accuracy'],
                "AUC": dkt_binary_domain['test_auc'],
                "F1 Score": dkt_binary_domain['test_f1']
            })
        
        if dkt_multiclass_domain:
            comparison_data.append({
                "Model": "DKT Multiclass",
                "Type": "Multiclass",
                "Primary Accuracy": dkt_multiclass_domain['test_accuracy'],
                "AUC": "-",
                "F1 Score": dkt_multiclass_domain['test_f1']
            })
        
        if comparison_data:
            df_comp = pd.DataFrame(comparison_data)
            st.dataframe(df_comp, use_container_width=True, hide_index=True)
            
            # Visualization
            st.markdown("### Accuracy Comparison (Domain Shift)")
            
            models = [d["Model"] for d in comparison_data]
            accuracies = [d["Primary Accuracy"] for d in comparison_data]
            
            fig = go.Figure(data=[
                go.Bar(x=models, y=accuracies,
                       marker_color=['#ffa520', '#813cf5', '#00c1f3'],
                       text=[f"{acc:.4f}" for acc in accuracies],
                       textposition='auto')
            ])
            fig.update_layout(
                title="Model Accuracy Comparison (Domain Shift)",
                yaxis_title="Accuracy",
                yaxis_range=[0, 1],
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
    
    else:  # Robustness Analysis
        st.markdown("### Model Robustness Analysis")
        st.info("Robustness is measured by the performance drop from Standard Test to Domain Shift Test")
        
        # Calculate robustness metrics
        robustness_data = []
        
        if hkt_standard and hkt_domain:
            acc_drop = hkt_standard['compilation_accuracy'] - hkt_domain['compilation_accuracy']
            auc_drop = hkt_standard['compilation_auc'] - hkt_domain['compilation_auc']
            f1_drop = hkt_standard['compilation_f1'] - hkt_domain['compilation_f1']
            robustness_data.append({
                "Model": "HKT",
                "Accuracy Drop": f"{acc_drop:.4f}",
                "AUC Drop": f"{auc_drop:.4f}",
                "F1 Drop": f"{f1_drop:.4f}",
                "Avg Drop": f"{(acc_drop + auc_drop + f1_drop)/3:.4f}"
            })
        
        if dkt_binary_standard and dkt_binary_domain:
            acc_drop = dkt_binary_standard['test_accuracy'] - dkt_binary_domain['test_accuracy']
            auc_drop = dkt_binary_standard['test_auc'] - dkt_binary_domain['test_auc']
            f1_drop = dkt_binary_standard['test_f1'] - dkt_binary_domain['test_f1']
            robustness_data.append({
                "Model": "DKT Binary",
                "Accuracy Drop": f"{acc_drop:.4f}",
                "AUC Drop": f"{auc_drop:.4f}",
                "F1 Drop": f"{f1_drop:.4f}",
                "Avg Drop": f"{(acc_drop + auc_drop + f1_drop)/3:.4f}"
            })
        
        if dkt_multiclass_standard and dkt_multiclass_domain:
            acc_drop = dkt_multiclass_standard['test_accuracy'] - dkt_multiclass_domain['test_accuracy']
            f1_drop = dkt_multiclass_standard['test_f1'] - dkt_multiclass_domain['test_f1']
            robustness_data.append({
                "Model": "DKT Multiclass",
                "Accuracy Drop": f"{acc_drop:.4f}",
                "AUC Drop": "N/A",
                "F1 Drop": f"{f1_drop:.4f}",
                "Avg Drop": f"{(acc_drop + f1_drop)/2:.4f}"
            })
        
        if robustness_data:
            df_robust = pd.DataFrame(robustness_data)
            st.dataframe(df_robust, use_container_width=True, hide_index=True)
            st.caption("Lower drop values indicate better robustness to domain shift")
            
            # Visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # Accuracy drop comparison
                models = [d["Model"] for d in robustness_data]
                acc_drops = [float(d["Accuracy Drop"]) for d in robustness_data]
                
                fig1 = go.Figure(data=[
                    go.Bar(x=models, y=acc_drops,
                           marker_color=['#e74c3c', '#e67e22', '#f39c12'],
                           text=[f"{drop:.4f}" for drop in acc_drops],
                           textposition='auto')
                ])
                fig1.update_layout(
                    title="Accuracy Drop (Standard → Domain Shift)",
                    yaxis_title="Accuracy Drop",
                    height=400
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Average drop comparison
                avg_drops = [float(d["Avg Drop"]) for d in robustness_data]
                
                fig2 = go.Figure(data=[
                    go.Bar(x=models, y=avg_drops,
                           marker_color=['#9b59b6', '#8e44ad', '#7d3c98'],
                           text=[f"{drop:.4f}" for drop in avg_drops],
                           textposition='auto')
                ])
                fig2.update_layout(
                    title="Average Performance Drop",
                    yaxis_title="Average Drop",
                    height=400
                )
                st.plotly_chart(fig2, use_container_width=True)

