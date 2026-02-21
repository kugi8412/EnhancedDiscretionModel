# ==============================================================================
# Shiny App: Data Exploration and Baseline Analysis
#
# DESCRIPTION:
# This Shiny application allows users to upload a dataset (CSV/TXT), select a 
# specific numeric variable (column), and perform interactive data cleaning 
# and exploratory data analysis. 
#
# CORE FUNCTIONALITIES:
# 1. Dynamic Column Selection: Calculates statistics and plots ONLY for the 
#    currently selected column.
# 2. Interactive Filtering: Users can filter values using a slider or remove 
#    specific outliers by selecting rows directly in the Data Table.
# 3. Baseline RMSE Calculation: Evaluates the Root Mean Square Error assuming 
#    the mean of the selected column is used as a naive predictor for all points.
# 4. Visualization: Plots the density and histogram of the selected variable, 
#    highlighting the Mean and Median with distinct, non-overlapping labels.
# ==============================================================================


# Cleaning environment
rm(list = ls())

# Load required libraries
suppressPackageStartupMessages({
  library(shiny)
  library(ggplot2)
  library(DT)
  library(dplyr)
  library(data.table)
})

# User Interface
ui <- fluidPage(
  titlePanel("Data Exploration and Baseline Analysis"),
  
  sidebarLayout(
    sidebarPanel(
      # File Upload
      h4("1. Upload Data"),
      fileInput("file1", "Choose file (CSV/TXT)",
                accept = c("text/csv", "text/comma-separated-values,text/plain", ".csv", ".txt")),
      radioButtons("sep", "Separator:",
                   choices = c("Comma" = ",", "Semicolon" = ";", "Tab" = "\t"),
                   selected = ","),
      hr(),
      
      # Variable Selection
      h4("2. Select Variable"),
      uiOutput("column_selector"),
      
      # Range Filtering
      h4("3. Filter Range"),
      uiOutput("slider_ui"),
      
      hr(),
      # Data Management (Row removal & Reset)
      h4("4. Operations"),
      p("Select rows in the table on the right to remove them."),
      actionButton("remove_rows", "Remove selected rows", class = "btn-danger"),
      actionButton("reset_data", "Reset all filters", class = "btn-warning"),
      br(), br(),
      downloadButton("download_data", "Download filtered data")
    ),
    
    mainPanel(
      tabsetPanel(
        tabPanel("Visualization & Statistics",
                 br(),
                 # Statistics Panel
                 wellPanel(
                   h4("Statistics (for current view)"),
                   fluidRow(
                     column(4, strong("Mean:"), textOutput("stat_mean", inline = TRUE)),
                     column(4, strong("Median:"), textOutput("stat_median", inline = TRUE)),
                     column(4, strong("N (sample size):"), textOutput("stat_n", inline = TRUE))
                   ),
                   hr(),
                   fluidRow(
                     column(12, 
                            strong("Baseline RMSE (if mean was the predictor):"), 
                            textOutput("stat_baseline", inline = TRUE),
                            p(style="font-size: 0.8em; color: gray;", 
                              "This is the error (RMSE) we would make by taking the mean as the prediction for all data points.")
                     )
                   )
                 ),
                 # Plot Downloads
                 fluidRow(
                   column(6, downloadButton("download_plot_png", "Download Plot (PNG)", class="btn-info", style="width:100%")),
                   column(6, downloadButton("download_plot_pdf", "Download Plot (PDF)", class="btn-info", style="width:100%"))
                 ),
                 br(),
                 plotOutput("distPlot", height = "400px")
        ),
        tabPanel("Data Table",
                 br(),
                 DTOutput("dataTable")
        )
      )
    )
  )
)

# Server
server <- function(input, output, session) {
  
  values <- reactiveValues(original = NULL, active = NULL, removed_indices = c())
  
  # File Upload
  observeEvent(input$file1, {
    req(input$file1)
    tryCatch({
      df <- read.csv(input$file1$datapath, sep = input$sep, stringsAsFactors = FALSE)
      df$row_id_internal <- 1:nrow(df)
      values$original <- df
      values$active <- df
      values$removed_indices <- c()
    }, error = function(e) {
      showNotification("Error loading file. Check the format.", type = "error")
    })
  })
  
  # Column Selection (Numeric only)
  output$column_selector <- renderUI({
    req(values$active)
    nums <- sapply(values$active, is.numeric)
    col_names <- names(nums)[nums]
    col_names <- col_names[col_names != "row_id_internal"]
    
    if (length(col_names) == 0) {
      return(p("No numeric columns found in the file."))
    }
    selectInput("selected_col", "Column to analyze:", choices = col_names)
  })
  
  output$slider_ui <- renderUI({
    req(values$active, input$selected_col)
    col_data <- values$active[[input$selected_col]]
    
    # Rounding min and max to prevent unreadable long decimals
    min_val <- round(min(col_data, na.rm = TRUE), 4)
    max_val <- round(max(col_data, na.rm = TRUE), 4)
    
    sliderInput("range_val", "Value range:",
                min = min_val, max = max_val,
                value = c(min_val, max_val))
  })
  
  final_data <- reactive({
    req(values$active, input$selected_col, input$range_val)
    
    df <- values$active
    col <- input$selected_col
    min_v <- input$range_val[1]
    max_v <- input$range_val[2]
    
    # Apply slider filter to the specifically selected column
    df_filtered <- df[df[[col]] >= min_v & df[[col]] <= max_v, ]
    return(df_filtered)
  })
  
  # Row Removal Logic
  observeEvent(input$remove_rows, {
    req(input$dataTable_rows_selected, values$active)
    current_view_data <- final_data()
    selected_internal_ids <- current_view_data$row_id_internal[input$dataTable_rows_selected]
    
    # Update "active" dataset by removing those specific IDs
    values$active <- values$active[!values$active$row_id_internal %in% selected_internal_ids, ]
    
    showNotification(paste("Removed", length(selected_internal_ids), "observations."), type = "warning")
  })
  
  # Data Reset
  observeEvent(input$reset_data, {
    req(values$original)
    values$active <- values$original
    showNotification("Data reset to initial state.", type = "message")
  })
  
  # Plot Generation Logic (Stored as reactive to reuse for plotting and downloading)
  current_plot <- reactive({
    req(final_data(), input$selected_col)
    df <- final_data()
    col_name <- input$selected_col
    val <- df[[col_name]]
    
    mean_val <- mean(val, na.rm = TRUE)
    median_val <- median(val, na.rm = TRUE)
    
    # Create a small dataframe for the lines to automatically generate a legend
    lines_df <- data.frame(
      Statistic = c("Mean", "Median"),
      Value = c(mean_val, median_val)
    )
    
    ggplot(df, aes_string(x = col_name)) +
      geom_histogram(aes(y = ..density..), fill = "lightblue", color = "black", bins = 30) +
      geom_density(alpha = 0.2, fill = "#FF6666") +
      # Solid, thick lines mapped by color to create a legend instead of messy text annotations
      geom_vline(data = lines_df, aes(xintercept = Value, color = Statistic), linewidth = 1.5, linetype = "solid") +
      scale_color_manual(name = "Statistic", values = c("Mean" = "blue", "Median" = "darkgreen")) +
      labs(title = paste("Distribution of variable:", col_name),
           y = "Density", x = "Value") +
      theme_minimal(base_size = 14)
  })
  
  # Distribution Plot Render
  output$distPlot <- renderPlot({
    current_plot()
  })
  
  # Plot Downloads
  output$download_plot_png <- downloadHandler(
    filename = function() { paste("distribution_plot_", Sys.Date(), ".png", sep = "") },
    content = function(file) {
      ggsave(file, plot = current_plot(), device = "png", width = 10, height = 6)
    }
  )
  
  output$download_plot_pdf <- downloadHandler(
    filename = function() { paste("distribution_plot_", Sys.Date(), ".pdf", sep = "") },
    content = function(file) {
      ggsave(file, plot = current_plot(), device = "pdf", width = 10, height = 6)
    }
  )
  
  # Statistics Calculations
  output$stat_mean <- renderText({
    req(final_data(), input$selected_col)
    val <- final_data()[[input$selected_col]]
    round(mean(val, na.rm = TRUE), 4)
  })
  
  output$stat_median <- renderText({
    req(final_data(), input$selected_col)
    val <- final_data()[[input$selected_col]]
    round(median(val, na.rm = TRUE), 4)
  })
  
  output$stat_n <- renderText({
    req(final_data())
    nrow(final_data())
  })
  
  output$stat_baseline <- renderText({
    req(final_data(), input$selected_col)
    val <- final_data()[[input$selected_col]]
    mean_val <- mean(val, na.rm = TRUE)
    # Calculate RMSE (Root Mean Square Error) for the baseline model
    mse <- mean((val - mean_val)^2, na.rm = TRUE)
    rmse <- sqrt(mse)
    
    round(rmse, 4)
  })
  
  output$dataTable <- renderDT({
    req(final_data())
    df_show <- final_data()
    df_show$row_id_internal <- NULL
    
    datatable(df_show, 
              options = list(pageLength = 10),
              selection = 'multiple')
  })
  
  # Data Download
  output$download_data <- downloadHandler(
    filename = function() {
      paste("filtered_data_", Sys.Date(), ".csv", sep = "")
    },
    content = function(file) {
      data_out <- final_data()
      data_out$row_id_internal <- NULL
      write.csv(data_out, file, row.names = FALSE)
    }
  )
}

# Run the application
shinyApp(ui = ui, server = server)
