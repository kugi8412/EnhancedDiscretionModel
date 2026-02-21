# ==============================================================================
# JASPAR 2024 Motif Scanner App
#
# DESCRIPTION:
# This Shiny application allows users to scan DNA sequences (FASTA) for 
# transcription factor binding sites (TFBS) using the JASPAR 2024 database.
# ==============================================================================


# Cleaning environment
rm(list = ls())

# Installing Packages
suppressPackageStartupMessages({
  library(shiny)
  library(shinyjs)
  library(Biostrings)
  library(TFBSTools)
  library(JASPAR2024)
  library(SummarizedExperiment)
  library(dplyr)
  library(DT)
  library(ggplot2)
  library(ggseqlogo)
  library(grid)
  library(BiocParallel)
  library(scales)
  library(reshape2)
})

# Upload limit for 1GB
options(shiny.maxRequestSize = 1024 * 1024^2)

# Species Definition
species_list <- list(
  "insects" = list("All Insects"="all", "Drosophila melanogaster"="7227", "Anopheles gambiae"="7165", "Apis mellifera"="7460"),
  "vertebrates" = list("All Vertebrates"="all", "Homo sapiens"="9606", "Mus musculus"="10090", "Danio rerio"="7955"),
  "fungi" = list("All Fungi"="all", "Neurospora crassa"="5141", "Saccharomyces cerevisiae"="4932", "Schizosaccharomyces pombe"="4896", "Aspergillus nidulans"="162425"),
  "plants" = list("All Plants"="all", "Arabidopsis thaliana"="3702", "Zea mays"="4577"),
  "nematodes" = list("All Nematodes"="all", "Caenorhabditis elegans"="6239"),
  "all" = list("All Database"="all")
)

global_worker_scanner <- function(i, mats, ids, names, seqs, seqs_rev, 
                                  scan_mode, strand_info, thr) {
  suppressPackageStartupMessages({
    library(stats); library(methods); library(Biostrings); library(Matrix)
  })
  
  pwm_matrix <- mats[[i]]
  hits_f_vec <- numeric(length(seqs))
  hits_r_vec <- numeric(length(seqs))
  
  if (scan_mode == "header") {
    idx_plus <- which(strand_info == "+")
    if(length(idx_plus) > 0) {
      hits_f_vec[idx_plus] <- suppressWarnings(sapply(seqs[idx_plus], function(s) countPWM(pwm_matrix, s, min.score = thr)))
    }
    idx_minus <- which(strand_info == "-")
    if(length(idx_minus) > 0) {
      hits_r_vec[idx_minus] <- suppressWarnings(sapply(seqs_rev[idx_minus], function(s) countPWM(pwm_matrix, s, min.score = thr)))
    }
  } else if (scan_mode == "both") {
    hits_f_vec <- suppressWarnings(sapply(seqs, function(s) countPWM(pwm_matrix, s, min.score = thr)))
    if(!is.null(seqs_rev)) {
      hits_r_vec <- suppressWarnings(sapply(seqs_rev, function(s) countPWM(pwm_matrix, s, min.score = thr)))
    }
  } else if (scan_mode == "fwd") {
    hits_f_vec <- suppressWarnings(sapply(seqs, function(s) countPWM(pwm_matrix, s, min.score = thr)))
  } else if (scan_mode == "rev") {
    if(!is.null(seqs_rev)) {
      hits_r_vec <- suppressWarnings(sapply(seqs_rev, function(s) countPWM(pwm_matrix, s, min.score = thr)))
    }
  }
  
  total_per_seq <- hits_f_vec + hits_r_vec
  total_sum <- sum(total_per_seq)
  
  if(total_sum > 0) {
    seqs_with_hits <- sum(total_per_seq > 0)
    return(data.frame(
      Motif_ID = ids[[i]],
      Motif_Name = names[[i]],
      Seqs_With_Hits = seqs_with_hits,
      Total_Hits = total_sum,
      Hits_Plus = sum(hits_f_vec),
      Hits_Minus = sum(hits_r_vec),
      Percentage = round(seqs_with_hits / length(seqs) * 100, 2)
    ))
  } else {
    return(NULL)
  }
}

# --- WORKER: PER-SEQUENCE STATS ---
sequence_level_worker <- function(seqs_chunk, pwms, thr, scan_mode) {
  suppressPackageStartupMessages({ library(Biostrings); library(TFBSTools) })
  
  n_seqs <- length(seqs_chunk)
  f_counts <- numeric(n_seqs)
  r_counts <- numeric(n_seqs)
  
  seqs_chunk_rev <- reverseComplement(seqs_chunk)
  strand_info <- rep("unknown", n_seqs)
  
  if(scan_mode == "header") {
    nm <- names(seqs_chunk)
    strand_info[grepl("_\\+_|\\+$", nm)] <- "+"
    strand_info[grepl("_\\-_|\\-$", nm)] <- "-"
  }
  
  for(i in seq_along(pwms)) {
    pwm <- pwms[[i]]
    if (scan_mode == "header") {
      idx_plus <- which(strand_info == "+")
      if(length(idx_plus) > 0) f_counts[idx_plus] <- f_counts[idx_plus] + suppressWarnings(sapply(seqs_chunk[idx_plus], function(s) countPWM(pwm, s, min.score = thr)))
      idx_minus <- which(strand_info == "-")
      if(length(idx_minus) > 0) r_counts[idx_minus] <- r_counts[idx_minus] + suppressWarnings(sapply(seqs_chunk_rev[idx_minus], function(s) countPWM(pwm, s, min.score = thr)))
    } else if (scan_mode == "both") {
      f_counts <- f_counts + suppressWarnings(sapply(seqs_chunk, function(s) countPWM(pwm, s, min.score = thr)))
      r_counts <- r_counts + suppressWarnings(sapply(seqs_chunk_rev, function(s) countPWM(pwm, s, min.score = thr)))
    } else if (scan_mode == "fwd") {
      f_counts <- f_counts + suppressWarnings(sapply(seqs_chunk, function(s) countPWM(pwm, s, min.score = thr)))
    } else if (scan_mode == "rev") {
      r_counts <- r_counts + suppressWarnings(sapply(seqs_chunk_rev, function(s) countPWM(pwm, s, min.score = thr)))
    }
  }
  
  return(data.frame(
    Sequence_ID = names(seqs_chunk),
    Length = width(seqs_chunk),
    Forward_Hits = f_counts,
    Reverse_Hits = r_counts,
    Total_Hits = f_counts + r_counts
  ))
}

# --- KROK 2: UI ---
ui <- fluidPage(
  useShinyjs(), 
  shiny::tags$head(shiny::tags$style(HTML("
    .modal-lg { width: 95% !important; }
    .stat-box { background-color: #ffffff; border-left: 5px solid #8e44ad; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.12); padding: 15px; text-align: center; margin-bottom: 20px; height: 100px; display: flex; flex-direction: column; justify-content: center; }
    .stat-num { font-size: 22px; font-weight: bold; color: #2c3e50; }
    .stat-label { font-size: 12px; color: #7f8c8d; text-transform: uppercase; letter-spacing: 1px; margin-top: 5px; }
    .bool-pass { color: #27ae60; font-weight: bold; font-size: 18px; }
    .bool-fail { color: #c0392b; font-weight: bold; font-size: 18px; }
    #open_jaspar_btn { background-color: #8e44ad; color: white; font-weight: bold; width: 100%; font-size: 16px; margin-top: 15px; border: none; }
    #open_jaspar_btn:hover { background-color: #9b59b6; }
    .seq-container { font-family: 'Courier New', monospace; font-size: 16px; letter-spacing: 2px; line-height: 2.0; white-space: pre; overflow-x: auto; background-color: #fdfdfd; padding: 20px; border: 1px solid #ccc; margin-top: 10px; }
    .nt-A { color: #27ae60; font-weight: 900; } 
    .nt-C { color: #2980b9; font-weight: 900; } 
    .nt-G { color: #f39c12; font-weight: 900; } 
    .nt-T { color: #c0392b; font-weight: 900; } 
    .nt-N { color: #95a5a6; }
  "))),
  
  titlePanel("JASPAR 2024 Motif Scanner"),
  
  sidebarLayout(
    sidebarPanel(
      width = 3,
      h4("1. Data Input"),
      fileInput("file_fasta", "Upload FASTA", accept = c(".fa", ".fasta", ".txt")),
      
      hr(),
      h4("2. Database"),
      selectInput("tax_group", "Taxonomy:", choices = c("Insects"="insects", "Vertebrates"="vertebrates", "Fungi"="fungi", "Plants"="plants", "Nematodes"="nematodes", "All Groups"="all"), selected = "insects"),
      selectInput("species_select", "Species:", choices = NULL),
      
      hr(),
      h4("3. Scanning Logic"),
      sliderInput("min_score", "Min Score (%):", min = 0, max = 100, value = 85, step = 1),
      selectInput("scan_mode_select", "Strand Scanning Mode:",
                  choices = c("Auto: Parse Headers (+/-)" = "header", "Global: Scan Both Strands" = "both", 
                              "Global: Forward Only (+)" = "fwd", "Global: Reverse Only (-)" = "rev"), selected = "both"),
      helpText("Auto: Uses header info to scan specific strands."),
      
      hr(),
      h4("4. Performance"),
      checkboxInput("use_parallel", "Parallel Processing", value = TRUE),
      checkboxInput("use_sampling", "Use Random Sample", value = FALSE),
      conditionalPanel(
        condition = "input.use_sampling == true",
        numericInput("sample_size", "Sample Size (N):", value = 1000, min = 100, step = 100)
      ),
      
      hr(),
      h4("5. Visualization"),
      fluidRow(
        column(8, numericInput("plot_top_n", "Top N Motifs:", value = 20, min = 5, max = 100)),
        column(4, actionButton("update_plot_btn", "Update", class = "btn-primary btn-sm", style = "margin-top: 25px;"))
      ),
      
      br(),
      actionButton("run_scan", "RUN SCAN", class = "btn-success icon-play", style="width: 100%; font-weight: bold;")
    ),
    
    mainPanel(
      width = 9,
      tabsetPanel(
        tabPanel("Quality Control",
                 br(),
                 h4("Global Dataset Statistics"),
                 fluidRow(
                   column(3, div(class="stat-box", div(class="stat-num", textOutput("qc_n_seqs")), div(class="stat-label", "Total Sequences"))),
                   column(3, div(class="stat-box", div(class="stat-num", textOutput("qc_avg_len")), div(class="stat-label", "Avg Length (bp)"))),
                   column(3, div(class="stat-box", uiOutput("qc_len_check"), div(class="stat-label", "Uniform Length?"))),
                   column(3, div(class="stat-box", uiOutput("qc_clean_check"), div(class="stat-label", "Clean DNA (No N)?")))
                 ),
                 br(),
                 fluidRow(
                   column(6, 
                          plotOutput("qc_plot_len", height="300px")
                   ), 
                   column(6, 
                          plotOutput("qc_plot_gc", height="300px"),
                          # DODANO: Przyciski pobierania dla wykresu GC
                          br(),
                          fluidRow(
                            column(6, downloadButton("dl_qc_gc_png", "Download GC Plot (PNG)", class="btn-xs btn-default", style="width:100%")),
                            column(6, downloadButton("dl_qc_gc_pdf", "Download GC Plot (PDF)", class="btn-xs btn-default", style="width:100%"))
                          )
                   )
                 )
        ),
        
        tabPanel("Global Results",
                 br(), h4("Global Motif Enrichment"),
                 DTOutput("motif_table"), br(),
                 downloadButton("download_results", "Download CSV", class = "btn-primary")
        ),
        
        tabPanel("Plots",
                 br(), plotOutput("motif_plot", height = "700px"), hr(),
                 fluidRow(
                   column(3, downloadButton("download_plot_png_top", "PNG (Top N)", class = "btn-info", style="width:100%")),
                   column(3, downloadButton("download_plot_pdf_top", "PDF (Top N)", class = "btn-info", style="width:100%")),
                   column(3, downloadButton("download_plot_png_all", "PNG (All)", class = "btn-warning", style="width:100%")),
                   column(3, downloadButton("download_plot_pdf_all", "PDF (All)", class = "btn-warning", style="width:100%"))
                 )
        ),
        
        tabPanel("Co-occurrence",
                 br(), h4("Motif Co-occurrence Correlation"),
                 plotOutput("corr_plot", height = "700px"), hr(),
                 fluidRow(column(4), column(2, downloadButton("download_corr_png", "PNG", class="btn-info", style="width:100%")), column(2, downloadButton("download_corr_pdf", "PDF", class="btn-info", style="width:100%")), column(4))
        ),
        
        tabPanel("Per-Sequence Stats",
                 br(), h4("Motif Counts Per Sequence"),
                 p("Calculates total hits for each sequence using the same sampling settings."),
                 actionButton("run_seq_stats", "Calculate Sequence Statistics", class = "btn-primary"),
                 br(), br(),
                 plotOutput("seq_stats_plot", height = "350px"), hr(),
                 DTOutput("seq_stats_table"), br(),
                 fluidRow(column(4, downloadButton("down_seq_stats_csv", "CSV", class="btn-success", style="width:100%")), column(4, downloadButton("down_seq_stats_png", "PNG", class="btn-info", style="width:100%")), column(4, downloadButton("down_seq_stats_pdf", "PDF", class="btn-info", style="width:100%")))
        ),
        
        tabPanel("Sequence Viewer",
                 br(), fluidRow(column(8, h4("Sequence Mapper")), column(4, uiOutput("batch_selector_ui"))),
                 DTOutput("seq_viewer")
        )
      )
    )
  )
)

# --- KROK 3: SERVER ---
server <- function(input, output, session) {
  
  current_motif_url <- reactiveVal(NULL)
  n_to_plot <- reactiveVal(20)
  
  observeEvent(input$update_plot_btn, { if (!is.na(input$plot_top_n) && input$plot_top_n > 0) n_to_plot(input$plot_top_n) })
  observe({ req(input$tax_group); updateSelectInput(session, "species_select", choices = species_list[[input$tax_group]]) })
  
  # 1. LOAD RAW DATA
  raw_data_loaded <- reactive({
    req(input$file_fasta)
    tryCatch({ readDNAStringSet(input$file_fasta$datapath) }, error = function(e) { showNotification("Error reading FASTA.", type="error"); NULL })
  })
  
  # 2. CREATE WORKING SET
  final_seq_set <- reactive({
    req(raw_data_loaded())
    raw <- raw_data_loaded()
    if(is.null(names(raw))) names(raw) <- paste0("Seq_", 1:length(raw))
    else names(raw) <- make.unique(names(raw))
    if (input$use_sampling && length(raw) > input$sample_size) {
      set.seed(123)
      return(raw[sample(length(raw), input$sample_size)])
    }
    return(raw)
  })
  
  # QC Outputs
  output$qc_n_seqs <- renderText({ req(final_seq_set()); format(length(final_seq_set()), big.mark=",") })
  output$qc_avg_len <- renderText({ req(final_seq_set()); round(mean(width(final_seq_set()))) })
  output$qc_len_check <- renderUI({ req(final_seq_set()); if(length(unique(width(final_seq_set()))) == 1) HTML("<span class='bool-pass'>YES <i class='fa fa-check'></i></span>") else HTML("<span class='bool-fail'>NO <i class='fa fa-times'></i></span>") })
  output$qc_clean_check <- renderUI({ req(final_seq_set()); if(!any(letterFrequency(final_seq_set(), "N")>0)) HTML("<span class='bool-pass'>YES <i class='fa fa-check'></i></span>") else HTML("<span class='bool-fail'>NO (Ns)</span>") })
  
  # QC Plots functions
  make_len_plot <- function(seqs) {
    lens <- width(seqs)
    if(length(unique(lens))==1) { 
      ggplot() + annotate("text",x=0.5,y=0.5,label=paste("Uniform:", lens[1],"bp"),size=8,color="#3498db") + theme_void() + theme(plot.title=element_text(hjust=0.5)) 
    } else { 
      ggplot(data.frame(L=lens), aes(x=L)) + geom_histogram(fill="#3498db", bins=30, color="white", alpha=0.8) + theme_classic() + labs(title="Length Dist", x="Length", y="Count") + theme(plot.title=element_text(hjust=0.5)) 
    }
  }
  
  # ZMIANA: Wykres GC dla wszystkich sekwencji (bez limitu 2000)
  make_gc_plot <- function(seqs) {
    freqs <- as.vector(letterFrequency(seqs, letters="GC", as.prob=TRUE))
    ggplot(data.frame(GC=freqs), aes(x=GC)) + geom_histogram(fill="#8e44ad", bins=30, color="white", alpha=0.8) + theme_classic() + labs(title="GC Content", x="GC Fraction", y="Count") + theme(plot.title=element_text(hjust=0.5))
  }
  
  output$qc_plot_len <- renderPlot({ req(final_seq_set()); make_len_plot(final_seq_set()) })
  output$qc_plot_gc <- renderPlot({ req(final_seq_set()); make_gc_plot(final_seq_set()) })
  
  # DOWNLOADS DLA QC GC
  output$dl_qc_gc_png <- downloadHandler(filename="qc_gc.png", content=function(file){ ggsave(file, plot=make_gc_plot(final_seq_set()), device="png", width=8, height=6) })
  output$dl_qc_gc_pdf <- downloadHandler(filename="qc_gc.pdf", content=function(file){ ggsave(file, plot=make_gc_plot(final_seq_set()), device="pdf", width=8, height=6) })
  
  # Motifs
  motifs_reactive <- reactive({
    req(input$species_select); opts <- list(collection = "CORE"); if (input$species_select != "all") opts[["species"]] <- input$species_select else if (input$tax_group != "all") opts[["tax_group"]] <- input$tax_group
    tryCatch({ db <- JASPAR2024(); if (!hasMethod("getMatrixSet", "JASPAR2024")) attr(db, "class") <- "JASPAR2022"; motifs <- getMatrixSet(db, opts); if(length(motifs)==0) { showNotification("No motifs found.", type="warning"); return(NULL) }; return(motifs) }, error = function(e) { NULL })
  })
  
  # 3. PAGINATION
  output$batch_selector_ui <- renderUI({ req(final_seq_set()); total <- length(final_seq_set()); bsize <- 100; n_batches <- ceiling(total/bsize); opts <- lapply(1:min(n_batches, 500), function(i) { list(name=paste0("Page ", i, " (", (i-1)*bsize+1, "-", min(i*bsize, total), ")"), value=i) }); selectInput("viewer_batch", NULL, choices=setNames(sapply(opts, function(x) x$value), sapply(opts, function(x) x$name)), selected=1, width="100%") })
  output$seq_viewer <- renderDT({ req(final_seq_set(), input$viewer_batch); all_seqs <- final_seq_set(); batch_idx <- as.numeric(input$viewer_batch); bsize <- 100; s_idx <- (batch_idx-1)*bsize+1; e_idx <- min(batch_idx*bsize, length(all_seqs)); subset <- all_seqs[s_idx:e_idx]; df <- data.frame(ID=names(subset), Length=width(subset), Sequence=as.character(subset)); df$Display_Seq <- ifelse(nchar(df$Sequence)>40, paste0(substr(df$Sequence, 1, 40), "..."), df$Sequence); datatable(df[,c("ID", "Length", "Display_Seq")], selection='single', rownames=FALSE, colnames=c("ID", "Length", "Sequence Preview"), options=list(pageLength=10, scrollX=TRUE)) })
  
  # 4. MAIN SCAN
  observeEvent(input$run_scan, { shinyjs::disable("run_scan"); showNotification("Scanning initialized...", type="message", duration=NULL, id="scan_init_msg") }, priority=2000)
  
  scan_data <- eventReactive(input$run_scan, {
    on.exit({ removeNotification("scan_init_msg"); shinyjs::enable("run_scan"); showNotification("Scan Finished!", type="message", duration=5) }, add=TRUE)
    req(final_seq_set(), motifs_reactive())
    
    used_seqs <- final_seq_set()
    used_seqs_rev <- reverseComplement(used_seqs)
    strand_info <- rep("unknown", length(used_seqs))
    scan_mode <- input$scan_mode_select
    if(scan_mode == "header") { nm <- names(used_seqs); strand_info[grepl("_\\+_|\\+$", nm)] <- "+"; strand_info[grepl("_\\-_|\\-$", nm)] <- "-" }
    
    motifs <- motifs_reactive(); if(is.null(motifs)) return(NULL)
    pwm_list <- toPWM(motifs); n_motifs <- length(pwm_list); thr <- paste0(input$min_score, "%")
    
    matrix_list_raw <- lapply(pwm_list, function(x) as.matrix(TFBSTools::Matrix(x)))
    id_list <- lapply(pwm_list, ID); name_list <- lapply(pwm_list, name)
    
    use_par <- input$use_parallel; if (length(used_seqs) < 2000) use_par <- FALSE
    
    withProgress(message=paste0('Processing ', length(used_seqs), ' sequences...'), value=0, {
      if(use_par) {
        if (.Platform$OS.type == "windows") BPPARAM <- SnowParam(workers=parallel::detectCores()-1, progressbar=TRUE) else BPPARAM <- MulticoreParam(workers=parallel::detectCores()-1, progressbar=TRUE)
        res_raw <- suppressWarnings(bplapply(seq_len(n_motifs), global_worker_scanner, mats=matrix_list_raw, ids=id_list, names=name_list, seqs=used_seqs, seqs_rev=used_seqs_rev, scan_mode=scan_mode, strand_info=strand_info, thr=thr, BPPARAM=BPPARAM))
        results_list <- do.call(rbind, res_raw)
      } else {
        results_list <- list()
        for(i in seq_len(n_motifs)) { if(i%%5==0) incProgress(1/n_motifs); res <- global_worker_scanner(i, matrix_list_raw, id_list, name_list, used_seqs, used_seqs_rev, scan_mode, strand_info, thr); if(!is.null(res)) results_list[[length(results_list)+1]] <- res }
        results_list <- do.call(rbind, results_list)
      }
    })
    
    if(is.null(results_list) || nrow(results_list)==0) return(NULL)
    final_df <- results_list %>% arrange(desc(Seqs_With_Hits))
    return(list(df=final_df, used_seqs=used_seqs, motifs_pfm=motifs))
  })
  
  # FIX: QUOTE = FALSE w downloadHandler
  output$motif_table <- renderDT({ req(scan_data()); datatable(scan_data()$df, selection='single', options=list(pageLength=10, order=list(2, 'desc')), rownames=FALSE) })
  output$download_results <- downloadHandler(filename = "results.csv", content = function(file) { write.csv(scan_data()$df, file, row.names=FALSE, quote=FALSE) })
  
  # --- SEQ STATS ---
  observeEvent(input$run_seq_stats, { shinyjs::disable("run_seq_stats"); showNotification("Calculating Stats...", type="message", duration=NULL, id="seq_stats_msg") }, priority=2000)
  
  seq_stats_data <- eventReactive(input$run_seq_stats, {
    on.exit({ removeNotification("seq_stats_msg"); shinyjs::enable("run_seq_stats"); showNotification("Stats Finished!", type="message", duration=5) }, add=TRUE)
    req(final_seq_set(), motifs_reactive()) # FIX: Używamy final_seq_set
    used_seqs <- final_seq_set()
    motifs <- motifs_reactive(); pwm_list <- toPWM(motifs); thr <- paste0(input$min_score, "%")
    matrix_list_raw <- lapply(pwm_list, function(x) as.matrix(TFBSTools::Matrix(x)))
    scan_mode <- input$scan_mode_select
    n_cores <- parallel::detectCores() - 1
    chunks <- split(used_seqs, cut(seq_along(used_seqs), breaks=n_cores, labels=FALSE))
    use_par <- input$use_parallel; if(length(used_seqs) < 2000) use_par <- FALSE
    
    withProgress(message = "Calculating per-sequence stats...", value = 0, {
      if(use_par) {
        if (.Platform$OS.type == "windows") BPPARAM <- SnowParam(workers=n_cores, progressbar=TRUE) else BPPARAM <- MulticoreParam(workers=n_cores, progressbar=TRUE)
        res_list <- suppressWarnings(bplapply(chunks, sequence_level_worker, pwms = matrix_list_raw, thr = thr, scan_mode = scan_mode, BPPARAM = BPPARAM))
        final_stats <- do.call(rbind, res_list)
      } else {
        res_list <- list()
        for(i in seq_along(chunks)) { incProgress(1/length(chunks)); res_list[[i]] <- sequence_level_worker(chunks[[i]], matrix_list_raw, thr, scan_mode) }
        final_stats <- do.call(rbind, res_list)
      }
    })
    final_stats$Density_Per_1kb <- round(final_stats$Total_Hits / final_stats$Length * 1000, 2)
    return(final_stats)
  })
  
  output$seq_stats_table <- renderDT({ req(seq_stats_data()); datatable(seq_stats_data(), options = list(pageLength=10, scrollX=TRUE), rownames=FALSE) }, server = TRUE)
  make_seq_stats_plot <- function(df) { ggplot(df, aes(x=Total_Hits)) + geom_histogram(fill="#8e44ad", color="white", bins=30, alpha=0.8) + theme_classic() + labs(title="Distribution of Total Motif Hits", x="Total Hits", y="Frequency") + theme(plot.title = element_text(hjust=0.5, face="bold")) }
  output$seq_stats_plot <- renderPlot({ req(seq_stats_data()); make_seq_stats_plot(seq_stats_data()) })
  
  # FIX: QUOTE = FALSE w downloadHandler
  output$down_seq_stats_csv <- downloadHandler(filename="seq_stats.csv", content=function(file){ write.csv(seq_stats_data(), file, row.names=FALSE, quote=FALSE) })
  output$down_seq_stats_png <- downloadHandler(filename="seq_stats.png", content=function(file){ ggsave(file, plot=make_seq_stats_plot(seq_stats_data()), device="png", width=10, height=6) })
  output$down_seq_stats_pdf <- downloadHandler(filename="seq_stats.pdf", content=function(file){ ggsave(file, plot=make_seq_stats_plot(seq_stats_data()), device="pdf", width=10, height=6) })
  
  # --- CORRELATION ---
  correlation_reactive <- reactive({ req(scan_data()); res <- scan_data(); top_n <- n_to_plot(); df_top <- head(res$df, top_n); top_ids <- df_top$Motif_ID; all_motifs <- res$motifs_pfm; pwm_list_top <- toPWM(all_motifs[top_ids]); seqs <- res$used_seqs; occ_matrix <- matrix(0, nrow=length(seqs), ncol=length(top_ids)); colnames(occ_matrix) <- paste0(df_top$Motif_Name, " (", df_top$Motif_ID, ")"); thr <- paste0(input$min_score, "%"); withProgress(message = "Calculating Correlation...", value=0, { for(i in 1:length(top_ids)) { incProgress(1/length(top_ids)); pwm <- as.matrix(TFBSTools::Matrix(pwm_list_top[[i]])); hits <- suppressWarnings(sapply(seqs, function(s) countPWM(pwm, s, min.score=thr))) + suppressWarnings(sapply(reverseComplement(seqs), function(s) countPWM(pwm, s, min.score=thr))); occ_matrix[, i] <- as.numeric(hits > 0) } }); cor(occ_matrix) })
  make_corr_plot <- function(cor_mat) { melted <- melt(cor_mat); ggplot(melted, aes(x=Var1, y=Var2, fill=value)) + geom_tile(color="white") + scale_fill_gradient2(low="blue", high="red", mid="white", midpoint=0, limit=c(-1,1), name="Pearson") + theme_minimal() + theme(axis.text.x=element_text(angle=45, vjust=1, size=10, hjust=1)) + coord_fixed() + labs(x="", y="", title="Co-occurrence") }
  output$corr_plot <- renderPlot({ req(correlation_reactive()); make_corr_plot(correlation_reactive()) })
  output$download_corr_png <- downloadHandler(filename="correlation.png", content=function(file){ ggsave(file, plot=make_corr_plot(correlation_reactive()), device="png", width=10, height=10) })
  output$download_corr_pdf <- downloadHandler(filename="correlation.pdf", content=function(file){ ggsave(file, plot=make_corr_plot(correlation_reactive()), device="pdf", width=10, height=10) })
  
  # --- PLOTS ---
  make_motif_plot <- function(data, title_suffix="") { data$Unique_Label <- paste0(data$Motif_Name, " (", data$Motif_ID, ")"); ggplot(data, aes(x=reorder(Unique_Label, Percentage), y=Percentage)) + geom_bar(stat="identity", fill="#8e44ad", width=0.7) + coord_flip(clip="off") + scale_y_continuous(expand=expansion(mult=c(0, 0.2))) + theme_classic(base_size=14) + labs(title=paste("Motif Occurrence", title_suffix), x="Motif", y="% Sequences") + geom_text(aes(label=paste0(Percentage, "%")), hjust=-0.1, size=3.5) + theme(plot.title=element_text(hjust=0.5, face="bold"), axis.text.y=element_text(color="black")) }
  output$motif_plot <- renderPlot({ req(scan_data()); make_motif_plot(head(scan_data()$df, n_to_plot()), paste("(Top", n_to_plot(), ")")) })
  output$download_plot_png_top <- downloadHandler(filename="top.png", content=function(file){ ggsave(file, plot=make_motif_plot(head(scan_data()$df, n_to_plot()), "(Top N)"), device="png", width=10, height=8) })
  output$download_plot_pdf_top <- downloadHandler(filename="top.pdf", content=function(file){ ggsave(file, plot=make_motif_plot(head(scan_data()$df, n_to_plot()), "(Top N)"), device="pdf", width=10, height=8) })
  output$download_plot_png_all <- downloadHandler(filename="all.png", content=function(file){ df<-scan_data()$df; h<-max(10, nrow(df)*0.35); ggsave(file, plot=make_motif_plot(df, "(All)"), device="png", width=12, height=h, limitsize=FALSE) })
  output$download_plot_pdf_all <- downloadHandler(filename="all.pdf", content=function(file){ df<-scan_data()$df; h<-max(10, nrow(df)*0.35); ggsave(file, plot=make_motif_plot(df, "(All)"), device="pdf", width=12, height=h, limitsize=FALSE) })
  
  observeEvent(input$motif_table_rows_selected, { req(scan_data()); row_idx <- input$motif_table_rows_selected; res <- scan_data(); row_data <- res$df[row_idx, ]; motif_id <- row_data$Motif_ID; motif_name <- row_data$Motif_Name; current_motif_url(paste0("https://jaspar.elixir.no/matrix/", motif_id, "/")); all_motifs <- res$motifs_pfm; pwm_raw <- as.matrix(TFBSTools::Matrix(toPWM(all_motifs[[motif_id]]))); thr <- paste0(input$min_score, "%"); sample_seqs <- res$used_seqs; if(length(sample_seqs)>2000) sample_seqs <- sample_seqs[sample(length(sample_seqs), 2000)]; strs_f <- unlist(lapply(suppressWarnings(lapply(sample_seqs, function(s) matchPWM(pwm_raw, s, min.score=thr))), as.character)); strs_r <- character(); if(input$scan_mode_select %in% c("both", "rev", "header")) { raw_r <- unlist(lapply(suppressWarnings(lapply(reverseComplement(sample_seqs), function(s) matchPWM(pwm_raw, s, min.score=thr))), as.character)); if(length(raw_r)>0) strs_r <- as.character(reverseComplement(DNAStringSet(raw_r))) }; showModal(modalDialog(title=paste("Motif Analysis:", motif_id, "-", motif_name), size="l", fluidRow(column(12, h4("Global Hit Statistics")), column(3, div(class="stat-box", div(class="stat-num", row_data$Seqs_With_Hits), div(class="stat-label", "Seqs with Hit"))), column(3, div(class="stat-box", div(class="stat-num", row_data$Total_Hits), div(class="stat-label", "Total Hits"))), column(3, div(class="stat-box", div(class="stat-num", row_data$Hits_Plus), div(class="stat-label", "Forward Hits"))), column(3, div(class="stat-box", div(class="stat-num", row_data$Hits_Minus), div(class="stat-label", "Reverse Hits")))), hr(), fluidRow(column(6, h4("Reference (Forward)"), plotOutput("modal_ref_logo", height="200px")), column(6, h4("Reference (Reverse)"), plotOutput("modal_ref_rev_logo", height="200px"))), hr(), p("Observed logos derived from hits found in your analyzed sequences (sample):"), fluidRow(column(6, h4("Observed (Native Data)"), plotOutput("modal_obs_f_logo", height="200px")), column(6, h4("Observed (Reverse)"), plotOutput("modal_obs_r_logo", height="200px"))), fluidRow(column(12, actionButton("open_jaspar_btn", "OPEN JASPAR DATABASE", icon=icon("external-link-alt")))), easyClose=TRUE)); output$modal_ref_logo <- renderPlot({ ggseqlogo(as.matrix(TFBSTools::Matrix(all_motifs[[motif_id]]))) + theme_minimal() }); output$modal_ref_rev_logo <- renderPlot({ ggseqlogo(as.matrix(TFBSTools::Matrix(reverseComplement(all_motifs[[motif_id]])))) + theme_minimal() }); output$modal_obs_f_logo <- renderPlot({ if(length(strs_f)<3) return(NULL); ggseqlogo(strs_f) + theme_minimal() }); output$modal_obs_r_logo <- renderPlot({ if(length(strs_r)<3) return(NULL); ggseqlogo(strs_r) + theme_minimal() }) })
  observeEvent(input$open_jaspar_btn, { req(current_motif_url()); browseURL(current_motif_url()) })
  
  observeEvent(input$seq_viewer_rows_selected, {
    req(scan_data(), final_seq_set(), input$viewer_batch) # FIX: Use final_seq_set
    sel_idx_local <- input$seq_viewer_rows_selected[1]; batch_idx <- as.numeric(input$viewer_batch); bsize <- 100; global_idx <- (batch_idx-1)*bsize + sel_idx_local; full_set <- final_seq_set(); if(global_idx > length(full_set)) return(NULL); target_seq <- full_set[[global_idx]]; target_name <- names(full_set)[global_idx]; seq_len <- length(target_seq); motifs <- motifs_reactive(); pwm_list <- toPWM(motifs); thr <- paste0(input$min_score, "%"); hits_df <- data.frame()
    withProgress(message=paste("Mapping ALL motifs on", target_name), value=0, { n_m <- length(pwm_list); step <- max(1, round(n_m/20)); for(i in seq_len(n_m)) { if(i%%step==0) incProgress(0.05); pwm <- as.matrix(TFBSTools::Matrix(pwm_list[[i]])); m_name <- name(pwm_list[[i]]); matches <- suppressWarnings(matchPWM(pwm, target_seq, min.score=thr, with.score=TRUE)); if(length(matches)>0) { for(j in seq_along(matches)) hits_df <- rbind(hits_df, data.frame(Motif=m_name, Start=start(matches)[j], End=end(matches)[j], Strand="+", Sequence=as.character(matches)[j])) }; matches_r <- suppressWarnings(matchPWM(pwm, reverseComplement(target_seq), min.score=thr, with.score=TRUE)); if(length(matches_r)>0) { for(j in seq_along(matches_r)) { s_r <- start(matches_r)[j]; e_r <- end(matches_r)[j]; g_start <- seq_len - e_r + 1; g_end <- seq_len - s_r + 1; s_c <- max(1, g_start); e_c <- min(seq_len, g_end); if(s_c <= e_c) { tryCatch({ seq_text <- as.character(subseq(target_seq, start=s_c, end=e_c)); hits_df <- rbind(hits_df, data.frame(Motif=m_name, Start=g_start, End=g_end, Strand="-", Sequence=seq_text)) }, error=function(e){}) } } } } })
    color_dna_safe <- function(seq_string) { chars <- strsplit(seq_string, "")[[1]]; replacements <- c("A"="<span class='nt-A'>A</span>", "C"="<span class='nt-C'>C</span>", "G"="<span class='nt-G'>G</span>", "T"="<span class='nt-T'>T</span>", "N"="<span class='nt-N'>N</span>"); mapped <- sapply(chars, function(char) { uc <- toupper(char); if(uc %in% names(replacements)) return(replacements[[uc]]) else return(char) }); return(paste(mapped, collapse="")) }; formatted_seq <- color_dna_safe(as.character(target_seq)); unique_motifs <- if(nrow(hits_df)>0) length(unique(hits_df$Motif)) else 0; plot_h <- max(300, 100 + unique_motifs * 30)
    showModal(modalDialog(title=paste("Sequence Map:", target_name, "(Length:", seq_len, "bp)"), size="l", h4("Motif Map (Independent Scan)"), uiOutput("dynamic_map_container"), hr(), h4("Full Sequence"), div(class="seq-container", HTML(formatted_seq)), easyClose=TRUE))
    output$dynamic_map_container <- renderUI({ plotOutput("seq_map_plot", height=paste0(plot_h, "px")) })
    output$seq_map_plot <- renderPlot({ if(is.null(hits_df) || nrow(hits_df)==0) return(ggplot()+annotate("text", x=1, y=1, label="No motifs detected.", size=6)+theme_void()); ggplot(hits_df, aes(xmin=Start, xmax=End, y=Motif, fill=Strand)) + geom_rect(aes(ymin=as.numeric(factor(Motif))-0.35, ymax=as.numeric(factor(Motif))+0.35), alpha=0.9) + scale_fill_manual(values=c("+"="#8e44ad", "-"="#e74c3c")) + scale_x_continuous(limits=c(1, seq_len), expand=c(0,0)) + theme_minimal() + labs(x="Position (bp)", y="Detected Motif") + theme(axis.text.y=element_text(size=10, face="bold"), panel.grid.major.y=element_line(color="grey90")) })
  })
}

shinyApp(ui, server)
