library(shiny)
library(shinydashboard)
library(DT)
library(readr)
library(dplyr)
library(tidyr)

data <- read_csv('/Users/sam_r/Downloads/expanded_data.csv', show_col_types = FALSE)

View(data)

data$year <- as.double(data$year)

data$week <- as.double(data$week)

years <- unique(data$year)

#data <- data.groupBy(data$year)

str(data)

ui <- dashboardPage(skin = "blue",
  dashboardHeader(title = "Betting Spreads in NFL games"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Spread_Dashboard", tabName = "spread_dashboard", icon = icon("dashboard")),
      menuItem("Over_Under_Dashboard", tabName = "over_under_dashboard", icon = icon("dashboard")),
      menuItem("Spread_Barchart_Dashboard", tabName = "spread_barchart_dashboard", icon = icon("dashboard")),
      menuItem("Over_Under_Barchart_Dashboard", tabName = "over_under_barchart_dashboard", icon = icon("dashboard")),
      menuItem("Data", tabName = "nfldata", icon = icon("football"))
    )
  ),
  dashboardBody(
    tabItems(
      tabItem("spread_dashboard",
        box(plotOutput("spread_plot"), width = 8),
        box(
          selectInput("spread_year", "Spread Year:",
                      choices = years), width = 4
        )
      ),
      tabItem("over_under_dashboard",
         box(plotOutput("over_under_plot"), width = 8),
         box(
           selectInput("over_under_year", "Over/Under Year:",
                       choices = years), width = 4
         )
      ),
      tabItem("spread_barchart_dashboard",
              box(plotOutput("spread_barplot"), width = 25),
              box(
                selectInput("bar_spread_year", "Spread Year:",
                            choices = years), width = 4
              )
      ),
      tabItem("over_under_barchart_dashboard",
              box(plotOutput("over_under_barplot"), width = 25),
              box(
                selectInput("bar_over_under_year", "Over/Under Year:",
                            choices = years), width = 4
              )
      ),
      tabItem("nfldata",
        fluidPage(
          h1("Spreads"),
          dataTableOutput("nfldata")
        )
      )
    )
  )
)

server <- function(input, output) {
  
  filter_spread_data <- reactive({
    #data %>% filter(data$year %in% input$year)
    #data <- subset(data, year == input$features)
    dplyr::filter(data, year == input$spread_year)
    #return(data)
  })
  
  output$spread_plot <- renderPlot({
    #plot(data$given_spread, data$actual_spread, xlab = "Given Spread", ylab = "Actual Spread", col=data$year)
    ggplot(filter_spread_data(), aes(given_spread, actual_spread)) + geom_point()
  })
  
  filter_over_under_data <- reactive({
    dplyr::filter(data, year == input$over_under_year)
    #return(data)
  })
  
  output$over_under_plot <- renderPlot({
    #plot(data$given_total, data$total_points, xlab = "Given Total", ylab = "Actual Total", col=data$year)
    ggplot(filter_over_under_data(), aes(given_total, total_points)) + geom_point()
  })
  
  filter_over_under_bardata <- reactive({
    data_2 <- select(data, year, week, given_total, total_points)
    dplyr::filter(data_2, year == input$bar_over_under_year) #%>% group_by(week)
    data_long_over_under <- data_2 %>% pivot_longer(cols=c(given_total, total_points), names_to = "Type", values_to = "Total")
    #return(data)
  })
  
  output$over_under_barplot <- renderPlot({
    #totals <- rep(c('given_total', 'total_points'))
    #value <- max(data$given_total, data$total_points)
    
    #plot(data$given_total, data$total_points, xlab = "Given Total", ylab = "Actual Total", col=data$year)
    ggplot(filter_over_under_bardata(), aes(x=week, y=Total, fill=Type)) + geom_bar(position="dodge", stat="identity")
  })
  
  filter_spread_bardata <- reactive({
    data_2 <- select(data, year, week, given_spread, actual_spread)
    dplyr::filter(data_2, year == input$bar_spread_year) #%>% group_by(week)
    data_long_spread <- data_2 %>% pivot_longer(cols=c(given_spread, actual_spread), names_to = "Type", values_to = "Total")
    #return(data)
  })
  
  output$spread_barplot <- renderPlot({
    #spreads <- rep(c('given_spread', 'actual_spread'))
    #value <- max(data$given_spread, data$actual_spread)
    #plot(data$given_total, data$total_points, xlab = "Given Total", ylab = "Actual Total", col=data$year)
    ggplot(filter_spread_bardata(), aes(x=filter_spread_bardata()$week, y=Total, fill=Type)) + geom_bar(position="dodge", stat="identity")
  })
  
  output$nfldata <- renderDataTable(data)
}

shinyApp(ui, server)