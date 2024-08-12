#' Plot Shock Data
#'
#' This function plots shock data along with average swimming velocity and
#' generates a bar graph showing the grand mean with confidence intervals.
#'
#' @param vec Numeric array of swimming velocity data rows are cells and colummns are video frames.
#' @param fps Numeric value indicating frames per second.
#' @param shock_on_times Numeric vector of shock on times in seconds.
#' @param shock_off_times Numeric vector of shock off times in seconds.
#'
#' @import sciplot
#'
#' @examples
#' filename<-system.file('data/out_15000_mV_rep03.csv', package='paramecium')
#' dat<-read.table(filename, sep=',', header=TRUE)
#' vec<-lapply(unique(dat$ID), function(x){sqrt(diff(dat$X[dat$ID==x])^2+diff(dat$Y[dat$ID==x])^2)  })
#' vec<-do.call("rbind", vec)
#'
#' logfile<-system.file('data/log_15000_mV_rep03.txt', package='paramecium')
#' log<-read.table(logfile, skip=1, header=TRUE, fill=TRUE, row.names = NULL)
#' names(log)<-c('Frame', 'Time', 'Stimuli', 'Event', 'Trial')
#'
#' fps<-30  # Replace with actual frames per second
#'
#' shock_on_times<-log$Frame[which(log$Stimuli=='US' & log$Event=='on')]/fps
#' shock_off_times<-log$Frame[which(log$Stimuli=='US' & log$Event=='off')]/fps
#'
#' out<-plot_shock_data(vec, fps, shock_on_times, shock_off_times)
#'
#' @export
plot_shock_data <- function(vec, fps, shock_on_times, shock_off_times, col='pink', event='Shock', remove=2, ymax=NULL) {
  layout(matrix(c(1,1,1,2,1,1,1,2), 2, 4, byrow = TRUE))
  speed<-(apply(vec, 2, mean, na.rm=TRUE))[-c(1:remove)]
  if(is.null(ymax)){
    ymax<-max((apply(vec, 2, mean, na.rm=TRUE))[-c(1:remove)])*1.2
  }

  plot((1:ncol(vec))[-c(1:remove)]/fps, (apply(vec, 2, mean, na.rm=TRUE))[-c(1:remove)],
       type = 'l', lwd = 1, ylim = c(0, ymax), ylab = "Mean swimming velocity", las = 1, xlab = "Time (seconds)")

  for (i in 1:length(shock_on_times)) {
    polygon(c(c(shock_on_times[i], shock_off_times[i]), rev(c(shock_on_times[i], shock_off_times[i]))), c(-10, -10, 20, 20), col = col)
  }

  box()
  lines((1:ncol(vec))[-c(1:remove)]/fps, (apply(vec, 2, mean, na.rm=TRUE))[-c(1:remove)])
  lines( smooth.spline((1:ncol(vec))[-c(1:remove)]/fps, (apply(vec, 2, mean, na.rm=TRUE))[-c(1:remove)]), col='red')

  ### extract perievent
  time<-(1:ncol(vec))[-c(1:remove)]/fps

  tim<-(1:ncol(vec))[-c(1:remove)]/fps
  shockon <- c()
  for (i in 1:length(shock_on_times)) {
    shockon[i] <- mean(speed[which(tim > shock_on_times[i] & tim < shock_off_times[i])], na.rm=TRUE)
    lines(c(shock_on_times[i], shock_off_times[i]), c(shockon[i], shockon[i]), lwd = 4, col = 'blue')
  }

  if(any(is.na(shockon))){
    shockon[is.na(shockon)]<-mean(shockon, na.rm = TRUE)
  }

  shockoff <- mean(speed[which(tim < shock_on_times[1])], na.rm=TRUE)
  lines(c(0, shock_on_times[1]), c(shockoff, shockoff), lwd = 4, col = 'blue')
  for (i in 2:length(shock_on_times)) {
    shockoff[i] <- mean(speed[which(tim < shock_on_times[i] & tim > shock_off_times[i-1])], na.rm=TRUE)
    lines(c(shock_off_times[i-1], shock_on_times[i]), c(shockoff[i], shockoff[i]), lwd = 4, col = 'blue')
  }
  shockoff[i+1] <- mean(speed[which(tim > shock_off_times[length(shock_off_times)])], na.rm=TRUE)
  lines(c(shock_off_times[i], max(tim)), c(shockoff[i+1], shockoff[i+1]), lwd = 4, col = 'blue')

  if(any(is.na(shockoff))){
    shockoff[is.na(shockoff)]<-mean(shockoff, na.rm = TRUE)
  }

  # Check if the sciplot package is installed, install it if necessary
  if (!requireNamespace("sciplot", quietly = TRUE)) {
    install.packages("sciplot")
  }

  library(sciplot)
  par(xpd = T)

  # Create the label vector for bar graph
  labels <- c(rep(paste('No', event), length(shockoff)),  rep(event, length(shockon)))

  # Calculate the mean for no-shock and shock conditions
  no_shock_mean <- shockoff
  shock_means <- shockon

  # Combine the means and calculate the relative means
  means <- c(no_shock_mean, shock_means)
  relative_means <- means / mean(no_shock_mean, na.rm=TRUE) * 100

  if(mean(shock_means, na.rm=TRUE)>mean(no_shock_mean, na.rm=TRUE)){
    ymax=120
  }else{
    ymax=102
  }

  sciplot::bargraph.CI(labels, relative_means, ylab = "Grand mean (%)", las = 1, ylim = c(0, ymax), xlab = '', col = c('white', col))
  par(xpd = F)

  # Create the list object with average velocity during shock and non-shock conditions
  output_list <- list(
    during_shock = shockon,
    non_shock = shockoff,
    perc.red = 100 - mean(shockon)/mean(shockoff) * 100
  )
  print(t.test(shockon, shockoff))
  return(output_list)

}



extract_trace <- function(vec, time, timestamp) {
  output_list <- vector("list", length(timestamp))

  for (i in seq_along(timestamp)) {
    event_time <- timestamp[i]
    start_time <- event_time - 2
    end_time <- event_time + 6

    # Find the indices corresponding to the start and end times
    start_index <- max(which(time <= start_time))
    end_index <- min(which(time >= end_time))

    # Extract the trace within the specified range
    trace <- vec[start_index:end_index]

    # Add the trace to the output list
    output_list[[i]] <- trace
  }

  return(output_list)
}
