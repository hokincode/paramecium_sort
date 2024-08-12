helper_function_viz_protocol<-function(df, exp, event = 'CS' ){
  index<-which(df$Ev==event & df$ent=='On' & df$exp==exp)
  polygon.start<-df$Time[index]/1000
  index<-which(df$Ev==event & df$ent=='Off' & df$exp==exp)
  polygon.end<-df$Time[index]/1000

  times<-list(start = polygon.start,
       end = polygon.end)

  return(times)
}

draw_polygon<-function(times, col='lightblue', height=0.25){

  height=c(-0.25, -0.25, height, height)

  out<-lapply(seq_along(times$start), function(x){
    polygon(c(times$start[x], times$end[x], times$end[x], times$start[x]),
            height, col=col)
  })

}

#' Visualize Protocol Data
#'
#' This function generates visualizations for the data frame containing protocol information.
#'
#' @param df A data frame containing protocol data.
#' @param exp which experiment to choose. Default is 0.
#' @param colCS color of the CS. Default is 'lightblue'.
#' @param colUS color of the US. Default is 'pink'.
#'
#' @details
#' This function takes a data frame 'df' as input and creates various visualizations
#' to help understand the protocol information.
#'
#' @examples
#' filename<-system.file('data/exp3.xlsx', package='paramecium')
#' df<-read_xlsx(path = filename)
#' viz_protocol(bank_df, main='My experiment', hUS=0.01)
#'
#' @keywords data visualization
#'
#' @export
viz_protocol <- function(df, main='My experiment', exp = 0, colCS = 'lightblue', colUS = 'pink', hCS = 0.25, hUS=0.125) {

  timesCS <- helper_function_viz_protocol(df, exp, event = 'CS')
  timesUS <- helper_function_viz_protocol(df, exp, event = 'US')

  plot(c(min(timesCS$start), 1.1*max(timesUS$end)), c(-1,1), type='n', ylab='', xlab='', axes=FALSE, main=main)

  draw_polygon(timesCS, col=colCS, height=hCS)

  draw_polygon(timesUS, col=colUS, height=hUS)

  arrows(min(df$Time)/1000, -0.25, x1 = 1.1*max(df$Time)/1000, y1 = -0.25, length = 0.25, angle = 30)

}
