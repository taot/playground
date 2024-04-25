import Vue from 'vue'
import App from './App'

import iView from 'iview'
import 'iview/dist/styles/iview.css'

Vue.use(iView);

new Vue({
    el: 'body',
    components: { App }
})
