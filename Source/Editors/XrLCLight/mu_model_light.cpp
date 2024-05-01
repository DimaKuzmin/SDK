#include "stdafx.h"

#include "mu_model_light.h"
#include "mu_model_light_threads.h"
#include "mu_light_net.h"
extern bool	mu_light_net = false;

 
void	wait_mu_base		()
{
	run_mu_base();
	wait_mu_base_thread		();
}