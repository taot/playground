<template>
    <h2>Test Buttons</h2>
    <i-button @click="info(1)">弹出通知提醒</i-button>
    <i-button type="primary" @click="info(0)">弹出通知提醒 (不自动关闭)</i-button>
    <i-button @click="load">
        <span v-if="!loading">开始加载</span>
        <span v-else>正在加载...</span>
    </i-button>
</template>

<script>
export default {
    data() {
        return {
            loading: false
        }
    },
    methods: {
        info(duration) {
            // this.$Message.info('Button clicked');
            console.log("duration: " + duration);
            this.$Notice.open({
                title: '哈哈哈',
                desc: '闲着没事就吃点饭',
                duration: duration
            });
        },
        load() {
            this.loading = true;
            var self = this;
            setTimeout(function() {
                self.loading = false;
            }, 3000);
        }
    }
}
</script>

<style>
</style>
